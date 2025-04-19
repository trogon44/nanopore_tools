from Bio import SeqIO, SeqRecord, Seq
import matplotlib.pyplot as plt
from Levenshtein import distance as lev_dist
import pysam
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import subprocess

def run_sys_command(command):
    print(f'Running command: {command}')
    result = subprocess.run(
        command, 
        shell=True, 
        stdout=subprocess.DEVNULL,  # Suppress stdout
        stderr=subprocess.PIPE      # Capture stderr, if needed
    )
    
    # Check the return code for success or failure
    if result.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Command failed. Error:", result.stderr.decode())
    return

class NanoporeMap():
    """"

    Class for mapping nanopore sequencing reads to a library of variants.

    Usage:

    nm = NanoporeMap(backbone_fasta, reads_fastq, varStart, varEnd)
    nm.fastq_info() (optional)
    nm.align2bakcbone()
    nm.set_library(library_fasta)
    nm.identify_library_variants()

    Description:
    This class has tools for mapping long-read sequencing data to members of a known variant library.
    The above usage is an example of the normal sequence

    Parameters
    -----------------------------------------------------
    makeplot: boolean, display scatterplot of length vs. mean q-score if True

    Returns
    -----------------------------------------------------
    None
    But it will build a dataframe of read length and average q-scores and a dictionary of
    summary statistics.

    """


    # class with reference fasta, fastq files of reads, and the start and stop indices of Cas protein
    def __init__(self, backbone_fasta, reads_fastq, varStart, varEnd, buffer = 5):
        self.backbone_fasta = Path(backbone_fasta)
        self.reads_fastq = Path(reads_fastq)
        self.varStart = varStart
        self.varEnd = varEnd
        self.fastq_index = SeqIO.index(str(self.reads_fastq), 'fastq')
        self.read_count = len(list(self.fastq_index.keys()))
        self.backbone_seqrecord = SeqIO.read(str(self.backbone_fasta), 'fasta')
        self.buffer = buffer

        print(f'Indexing fastq: "{str(self.reads_fastq)}"')
        print(f'Fastq contains {self.read_count} reads.')

    # generate fastq info dataframe (and plot jointplot if desired)
    def fastq_info(self, makeplot=True):
        """"

        Get some info about the fastq dataset.

        Usage:
        fastq_info(makeplot=True)

        Description:
        This gets some information on the fastq read dataset and includes a visualization
        of read length and quality if makeplot=True

        Parameters
        -----------------------------------------------------
        makeplot: boolean, display scatterplot of length vs. mean q-score if True

        Returns
        -----------------------------------------------------
        None
        But it will build a dataframe of read length and average q-scores and a dictionary of
        summary statistics.

        """
        read_info = dict()
        for read in self.fastq_index.keys():
            record = self.fastq_index[read]
            read_info[read] = {'length': len(record.seq), 'mean_q': np.mean(record.letter_annotations['phred_quality'])}
        self.fastqinfo = pd.DataFrame.from_dict(read_info, orient='index')

        if makeplot:
            sns.jointplot(data=self.fastqinfo, x='length', y='mean_q', s=5)
            plt.show()

        self.fastq_summary = {'total reads': len(self.fastqinfo),
                            'avg. length': np.mean(self.fastqinfo['length']),
                            'avg. q-score': np.mean(self.fastqinfo['mean_q'])}
        
        return

    # filter reads according to selection criteria and create pairwise alignment dictionary with minimap2
    def align2backbone(self, backbone_qscore_cutoff = 20,
                       min_length = 1000,
                       return_seqs = False,
                       var_region_qscore_cuttoff = 30,
                       overwrite_var_region = True,
                       double_plasmid = True):
        """"

        Aligns fastq sequences to a backbone plasmid fasta.

        Usage:
        align2backbone(qscore_cutoff = 20,
                       min_length = 1000, 
                       buffer = 5, 
                       return_seqs = False, 
                       var_region_qscore_cuttoff = 30, 
                       overwrite_var_region = True,
                       double_plasmid = True)

        Description:
        This function aligns the fastq reads to the backbone fasta and extracts
        the sequence matching the variable region +/- the specified buffer zone
        and writes a fastq.
        If return_seqs is True it will return a list of biopython seqrecords of
        the sequences in the fastq.

        Parameters
        -----------------------------------------------------
        qscore_cutoff (float, default 20): minimum average read Q-score allowed
        min_length (int, default 1000): minimum allowed read length
        buffer (int, default 5): amount of buffer sequence on either side of the variable region,
        you might need this to be longer if there is a lot of different sizes of the variable region
        return_seqs (boolean, default False)
        var_region_qscore_cuttoff (float, default 30): minimum average Q-score allowed for the variable region
        overwrite_var_region (boolean, default True): if True, overwrites the variable region of the backbone fasta with N's
        double_plasmid (boolean, default True): if True, doubles the backbone fasta sequence to account for the circular nature of the plasmid and random tagmentation insertion sites.
        This is only necessary if your sample is an adapter-ligated library.
        If False, the backbone fasta will be used as is.

        Returns
        -----------------------------------------------------
        list of biopython seqrecords if return_seqs = True

        """
        if hasattr(self, 'varDelta'):
            print("This object has already been aligned to the backbone. If you want to re-align, please create a new object.")
            return
        
        # This is the index tracker that records modifications to the variable region start and stop indices for the backbone sequence
        self.varDelta = 0
        
        # check if fastqinfo exists. if not, create it.
        if not hasattr(self, 'fastqinfo'):        
            self.fastq_info(makeplot=False)

            
        filter_keys = self.fastqinfo[(self.fastqinfo['length'] > min_length) & (self.fastqinfo['mean_q'] > backbone_qscore_cutoff)].index
        filter_records = [self.fastq_index[k] for k in filter_keys]
        print(f"{len(filter_records)}/{self.read_count} reads passed length and quality checks.")

        temp_fastq = self.reads_fastq.parent / 'temp.fastq'
        SeqIO.write(filter_records, temp_fastq, 'fastq')

        # Overwrite variable region with N's and align to this instead.
        if overwrite_var_region:
            print("Writing new backbone plasmid with variable region overwritten as N's.")
            overwrite_seq = Seq.Seq(''.join(['N' if ((ind >= self.varStart - 1) and (ind <= self.varEnd - 1)) else n for ind, n in enumerate(self.backbone_seqrecord.seq)]))
            new_backbone_fasta = self.backbone_fasta.parent / (self.backbone_fasta.stem + '_Ns' + self.backbone_fasta.suffix)
            new_backbone_seqrecord = SeqRecord.SeqRecord( name =  self.backbone_seqrecord.name + "_Ns",
                                            id = self.backbone_seqrecord.name + "_Ns",
                                            seq = overwrite_seq,
                                            description = "backbone vector with variable region as Ns")

            self.backbone_fasta = new_backbone_fasta
            self.backbone_seqrecord = new_backbone_seqrecord

            if not double_plasmid:
                SeqIO.write(self.backbone_seqrecord, self.backbone_fasta, 'fasta')
                print(f"New backbone fasta written as: {self.backbone_fasta}")

        if double_plasmid:
            print("Writing new backbone fasta with duplicated non-variable regions.")
            print("This is to account for the circular nature of the plasmid and random tagmentation insertion sites.")
            print("If your sample is an adapter-ligated library, you will want to set double_plasmid = False.")
            doubled_seq = Seq.Seq(self.backbone_seqrecord.seq[self.varEnd:] + self.backbone_seqrecord.seq + self.backbone_seqrecord.seq[:self.varStart:])
            new_backbone_seqrecord = SeqRecord.SeqRecord( name =  self.backbone_seqrecord.name + "_doubled",
                                            id = self.backbone_seqrecord.name + "_doubled",
                                            seq = doubled_seq,
                                            description = "backbone vector doubled outside of variable region")
            new_backbone_fasta = self.backbone_fasta.parent / (self.backbone_fasta.stem + '_doubled' + self.backbone_fasta.suffix)
            self.backbone_fasta = new_backbone_fasta
            self.varDelta = len(self.backbone_seqrecord.seq[self.varEnd:])
            self.backbone_seqrecord = new_backbone_seqrecord

            SeqIO.write(self.backbone_seqrecord, self.backbone_fasta, 'fasta')
            
            print(f"New backbone fasta written as: {self.backbone_fasta}")
                 


        temp_align = self.reads_fastq.parent / 'temp.sam'
        minimap2_call = f'minimap2 -ax map-ont {self.backbone_fasta} {temp_fastq} > {temp_align}'
        
        print(minimap2_call)
        stderr = os.system(minimap2_call)
        # exit with error if minimap2 returns abnormally
        if stderr != 0:
            raise OSError("Minimap2 command failed with abnormal exit status.")

        print("Finished with alignment.")

        seqrecords = []
        counter = np.zeros(6)
        with pysam.AlignmentFile(temp_align, "r") as alignments:
            aname = ''
            for alignment in alignments:
                counter[0] += 1
                # check if the read is mapped and primary; if not, skip it
                if alignment.is_unmapped or alignment.is_secondary or alignment.is_supplementary:
                    counter[1] += 1
                    continue
                # make sure the read covers the variable region
                covers_variable_region = (alignment.reference_start < self.varStart + self.varDelta) and (alignment.reference_end > self.varEnd + self.varDelta)
                if not covers_variable_region:
                    counter[2] += 1
                    continue            

                # This checks to see whether the read already has an entry. In principle this should be redundant with primary mapped        
                name = alignment.query_name
                if name == aname:
                    counter[3] += 1
                    continue
                else:
                    aname = name

                # Get the pairwise dictionary for the alignment
                pairwise = np.array(alignment.get_aligned_pairs(matches_only=True))

                # Get the coordinates of the variable region in the query
                try:
                    query_var_start = pairwise[np.where(pairwise[:,1] < (self.varStart + self.varDelta - self.buffer))[0][-1],0]
                    query_var_end = pairwise[np.where(pairwise[:,1] > (self.varEnd +self.varDelta + self.buffer))[0][0],0]
                except:
                    counter[4] += 1
                    continue

                # Get the q-scores for the variable region
                quality_scores = [alignment.query_qualities[i] for i in np.arange(query_var_start, query_var_end, 1)]
                # check if the q-scores are above the cutoff and discard if not
                if np.mean(quality_scores) < var_region_qscore_cuttoff:
                    counter[5] += 1
                    continue
                    
                # Build the SeqRecord object for the variable region
                seqrecord = SeqRecord.SeqRecord( name =  name,
                                                id = name,
                                                seq = Seq.Seq(alignment.query_sequence[query_var_start:query_var_end]),
                                                description = "aligned to variable region",
                                                letter_annotations={"phred_quality": quality_scores})

                seqrecords.append(seqrecord)
        self.temp_align_fastq = self.reads_fastq.parent / 'temp_aligned.fastq'
        SeqIO.write(seqrecords, self.temp_align_fastq, 'fastq')
        print(f"{len(seqrecords)}/{len(filter_records)} aligned to backbone and extracted to {self.temp_align_fastq}")
        self.counter = counter
        if return_seqs:
            return seqrecords


    def set_library(self, library_fasta):
        """"

        Assigns library fasta file for alignment and writes library fasta with buffer flanks.

        Usage:
        set_library(library_fasta)

        Returns
        -----------------------------------------------------
        None

        """
        self.library_fasta = library_fasta
        print(f'Variants added {library_fasta} added as variant list.')
        design_records = []
        
        try:
            left_flank = self.backbone_seqrecord.seq[self.varStart - self.buffer: self.varStart]
            right_flank = self.backbone_seqrecord.seq[self.varEnd: self.varEnd + self.buffer]
        except:
            print(f"Buffer region impinges on the backbone sequence margin.")
            print(f"Please change buffer and re-run align2backbone.")
            return

        self.library_fasta_flank = f'library_{self.buffer}flank.fasta'
        for drec in SeqIO.parse(self.library_fasta, 'fasta'):
            drec_seq = left_flank + drec.seq + right_flank
            drec_append = SeqRecord.SeqRecord(name = drec.name,
                                                id = drec.id,
                                                description = f"with {self.buffer} buffer flanks",
                                                seq = drec_seq)
            design_records.append(drec_append)
        SeqIO.write(design_records, self.library_fasta_flank, 'fasta')

    def identify_library_variants(self, hit_number = 2, lev_calc = False, use_fasta = False, fasta_name = ''):
        if not hasattr(self, 'library_fasta_flank'):
            print("Must run set_library(library_fasta) first before identifying library variants.")
            print("Exiting with no action")
            return

        
        if (not hasattr(self, "temp_align_fastq")) and not use_fasta:
            print("Must run align2backbone() first to get aligned reads.")
            print("Exiting with no action")
            return

        if use_fasta and (fasta_name == ''):
            print("Please provide a name for the fasta as fasta_name = 'some_name.fasta'")
            return

        temp_align2designs = 'align_to_designs.sam'


        if use_fasta:
            seq_records = []
            temp_align_fasta = 'temp_align.fasta'
            for rec in SeqIO.parse(fasta_name, 'fasta'):
                seq_record = SeqRecord.SeqRecord(id = rec.id, name = rec.name, description = f"aligned with {self.buffer} buffer flanks", seq = rec.seq[self.varStart - self.buffer: self.varEnd + self.buffer])
                seq_records.append(seq_record)
            SeqIO.write(seq_records, temp_align_fasta, 'fasta')


            minimap2_call = f"minimap2 -ax map-ont {self.library_fasta_flank} {temp_align_fasta} > {temp_align2designs}"
        else:
            minimap2_call = f"minimap2 -ax map-ont {self.library_fasta_flank} {self.temp_align_fastq} > {temp_align2designs}"

        print(minimap2_call)
        stderr = os.system(minimap2_call)
        # exit with error if minimap2 returns abnormally
        if stderr != 0:
            raise OSError("Minimap2 command failed with abnormal exit status.")
        

        # Initialize the dictionary to store matches
        matches_dict = defaultdict(lambda: {f'match_{i+1}': None for i in range(hit_number)})

        print(f"Reading minimap2 alignment and gathering first {hit_number} matches, if they exist.")
        # Open the SAM/BAM file
        with pysam.AlignmentFile(temp_align2designs, "r") as alignments:
            for alignment in alignments:
                if alignment.is_mapped:
                    query_name = alignment.query_name
                    ref_name = alignments.get_reference_name(alignment.reference_id)

                    for i in range(hit_number):
                        match_key = f'match_{i+1}'
                        if matches_dict[query_name][match_key] is None:
                            matches_dict[query_name][match_key] = ref_name
                            break

        self.match_df = pd.DataFrame.from_dict(matches_dict, orient='index')

        if lev_calc:
            ld_cols = [f"{col}_lev_dist" for col in self.match_df.columns]
            ld_dict = dict()
            
            if use_fasta:
                mapped_index = SeqIO.index(temp_align_fasta, 'fasta')
            else:
                mapped_index = SeqIO.index(self.temp_align_fastq, 'fastq')
            lib_fasta_index = SeqIO.index(self.library_fasta_flank, 'fasta')
            for row in self.match_df.iterrows():
                read_seq = mapped_index[row[0]].seq
                ldist = [lev_dist(read_seq.upper(), lib_fasta_index[rdesign].seq.upper()) if rdesign != None else np.nan for rdesign in row[1]]
                ld_dict[row[0]] = {j:i for i,j in zip(ldist, ld_cols)}

            levdf = pd.DataFrame.from_dict(ld_dict, orient='index')
            levdf = levdf.astype(pd.Int32Dtype())
            self.match_df = self.match_df.join(levdf)

        unique_designs = np.unique(self.match_df['match_1'], return_counts=True)
        unique_designs_sort = np.argsort(unique_designs[1])[::-1]
        self.match_summary = pd.DataFrame({'read_counts': unique_designs[1][unique_designs_sort]}, index=unique_designs[0][unique_designs_sort])

    def pop_seqs(self, id_list0, lib_or_reads = 'reads', write_fasta = False, fasta_name = ''):

        """"

        Returns a dictionary of biopython seqrecords of the variable regions for reads with ids in id_list0.
        id_list0 can be either a list or a string of just one entry.
        Optional can ouput a fasta file of the variable region sequences with write_fasta = True and fasta_name = 'some_name.fasta'

        Usage:
        pop_seqs(id_list0)

        Returns
        -----------------------------------------------------
        Dictionary of seqrecords with the key being the read id (basically like SeqIO.index)

        """
        if write_fasta and (fasta_name == ''):
            print("You must provide a name for the fasta file, ie. pop_seqs(id_list0, write_fasta = True, fasta_name = 'some_name.fasta')")
            return


        if type(id_list0) == str:
            id_list = [id_list0]
        else:
            id_list = id_list0.copy()
        
        seq_records_dict = dict()

        if lib_or_reads == 'reads':
            for rec in SeqIO.parse(self.temp_align_fastq, 'fastq'):
                if rec.id in id_list:
                    seq_records_dict[rec.id] = rec

        elif lib_or_reads == 'lib':
            for rec in SeqIO.parse(self.library_fasta_flank, 'fasta'):
                if rec.id in id_list:
                    seq_records_dict[rec.id] = rec

        else:
            print("Please specify if the list of sequence ids are library variants ('lib') or sequencing reads ('reads').")
            print("Exiting with no action.")
            return          

        if len(seq_records_dict) == 0:
            print("Didn't find records matching the provided ids. Are you sure that 'lib' or 'reads' is consistent with the id list?")
            return

        elif len(seq_records_dict) < len(id_list):
            unused_ids = list(set(id_list) - set(list(seq_records_dict.keys())))
            print(f"There are {len(unused_ids)} unused ids.")
            
            for ind, ui in enumerate(unused_ids):
                print(ui)
                if ind > 10:
                    print("etc.")
                    break

        if write_fasta:
            seq_keys = list(seq_records_dict.keys())
            seq_records = [seq_records_dict[k] for k in seq_keys]
            SeqIO.write(seq_records, fasta_name, 'fasta')

        return seq_records_dict        


    # def extract_barcodes(self, barStart, barEnd):
    #     # get read ids for all reads that were mapped to variants.
    #     mapped_read_ids = self.match_df.index
    #     # write fastq of only these.
    #     mapped_read_records = [self.fastq_index[mri] for mri in mapped_read_ids]
    #     temp_mapped_read_fastq = 'temp_mapped_read.fastq'
    #     SeqIO.write(mapped_read_records, temp_mapped_read_fastq, 'fastq')
        
    #     # # make reference fasta for barcode (with 50 buffer on either side).
    #     # barcode_flank_seq = self.backbone_fasta.seq[barStart - 50: barEnd + 50]
    #     # barcode_seqrec = SeqRecord.SeqRecord(id = 'barcode_flank', name = 'barcode_flank', description = "barcode +/- 50", seq = barcode_flank_seq)
    #     # SeqIO.write

class LibVarCount():
    """"

    Class for mapping nanopore sequencing reads to a library of variants.

    Usage:

    lvc = LibVarCount(lib_fasta, reads_fastq)
    lvc.fastq_info() (optional)
    lvc.count_mapped_variants()
    lvc.plot_hits()
    lvc.write_hit_fastqs(variant, number, outname0)

    Description:
    This class has tools for mapping long-read sequencing data to members of a known variant library.
    The above usage is an example of the normal sequence

    Arguments
    -----------------------------------------------------
    lib_fasta: path to fasta file with library variants
    reads_fastq: path to fastq file with nanopore reads

    """
    def __init__(self, lib_fasta, reads_fastq):
        self.lib_fasta = Path(lib_fasta)
        self.reads_fastq = Path(reads_fastq)
        self.workdir = self.reads_fastq.parent
        self.fastq_index = SeqIO.index(str(self.reads_fastq), 'fastq')
        self.read_count = len(list(self.fastq_index.keys()))
    
        print(f'Indexing fastq: "{str(self.reads_fastq)}"')
        print(f'Fastq contains {self.read_count} reads.')
    
    def fastq_info(self, makeplot=True):
        """"

        Get some info about the fastq dataset.

        Usage:
        fastq_info(makeplot=True)

        Description:
        This gets some information on the fastq read dataset and includes a visualization
        of read length and quality if makeplot=True

        Parameters
        -----------------------------------------------------
        makeplot: boolean, display scatterplot of length vs. mean q-score if True

        Returns
        -----------------------------------------------------
        None
        But it will build a dataframe of read length and average q-scores and a dictionary of
        summary statistics.

        """    
        read_info = dict()
        for read in self.fastq_index.keys():
            record = self.fastq_index[read]
            read_info[read] = {'length': len(record.seq), 'mean_q': np.mean(record.letter_annotations['phred_quality'])}
        self.fastqinfo = pd.DataFrame.from_dict(read_info, orient='index')
    
        if makeplot:
            sns.jointplot(data=self.fastqinfo, x='length', y='mean_q', s=5)
            plt.show()
    
        self.fastq_summary = {'total reads': len(self.fastqinfo),
                            'avg. length': np.mean(self.fastqinfo['length']),
                            'avg. q-score': np.mean(self.fastqinfo['mean_q'])}
        return

    
    def get_align_filename(self, align_name0):
        # for internal use.
        # assigns the alignment file name in the working directory
        if Path(align_name0).parent != self.workdir:
            align_name = self.workdir / (Path(align_name0).stem + ".sam")
        else:
            align_name = align_name0
        self.align_name = align_name
        return
    
    def minimap_align(self):
        # for internal use
        # runs minimap2 to align the reads against the fasta library
        minimap2_call = f'minimap2 -ax map-ont {str(self.lib_fasta)} {str(self.reads_fastq)} > {str(self.align_name)}'
        print(f'Running: {minimap2_call}')
        # Execute the command and suppress stdout
        result = subprocess.run(
            minimap2_call, 
            shell=True, 
            stdout=subprocess.DEVNULL,  # Suppress stdout
            stderr=subprocess.PIPE      # Capture stderr, if needed
        )
        
        # Check the return code for success or failure
        if result.returncode == 0:
            print("Command executed successfully.")
        else:
            print("Command failed. Error:", result.stderr.decode())
        print(f'Writing alignment to {str(self.align_name)}')
        
        return
    
    def analyze_align(self):
        # for internal use
        # parses the alignment file and builds a dataframe with all mapped reads
        align_dict = {}
        with pysam.AlignmentFile(self.align_name, "r") as alignments:
            for alignment in alignments:
                if alignment.is_mapped:
                    ref_name = alignment.reference_name
                    query_name = alignment.query_name
                    map_qual = alignment.mapping_quality
                    overlap = alignment.query_alignment_length
                    align_dict[query_name] = {'ref_name':ref_name, 'map_quality':map_qual, 'overlap_length':overlap}
        self.align_df = pd.DataFrame.from_dict(align_dict, orient='index')
        return
    
    def pop_fast(self, fast_ids, outname0, keepinds = True):
        # exports fasta or fastq files from read ids
        # check if fastqinfo exists. if not, create it.
        if not hasattr(self, 'fastqinfo'):
            self.fastq_info(makeplot=False)
        
        if Path(outname0).parent != self.workdir:
            outname = self.workdir / (Path(outname0).stem + Path(outname0).suffix)
        else:
            outname = outname0
    
        
        seqrecords = []
        for ind, fid in enumerate(fast_ids):
            try:
                seqrecord = self.fastq_index[fid]
                if keepinds:
                    nameind = fid
                else:
                    nameind = f'pop{ind}'
                    
                seqrecord.id = nameind
                seqrecord.name = nameind
                seqrecord.description = ''
                
                seqrecords.append(seqrecord)
            except:
                print(f'{fid} not found in index.')
        if (Path(outname).suffix == '.fa') | (Path(outname).suffix == '.fasta'):
            filetype = 'fasta'
        elif (Path(outname).suffix == '.fq') | (Path(outname).suffix == '.fastq'):
            filetype = 'fastq'
        else:
            raise ValueError(f'Unknown extension for {outname}. Must be fasta or fastq.') 
        print(f"Writing {len(seqrecords)} as {filetype} to {outname}")
        SeqIO.write(seqrecords, outname, filetype)
        return
    
    def count_hits(self):
        # counts occurance of variants and builds a sorted dataframe of them
        unique_hits = np.unique(self.align_df['ref_name'], return_counts=True)
        ordered_ind = np.argsort(unique_hits[1])[::-1]
        self.hits_df = pd.DataFrame({'counts': unique_hits[1][ordered_ind]}, index=unique_hits[0][ordered_ind])
        return

    def count_mapped_variants(self):
        """"

        Maps the reads to the variants and compiles statistics.

        Usage:
        lvc.count_mapped_variants()

        Description:
        This performs a minimap2 alignment of the fastq reads against the fasta
        of library variants. It counts up the mapping occurances.

        -----------------------------------------------------
        None
        But creates lvc.align_df, a dataframe with information about all mapped reads,
        and lvc.hits_df, a dataframe with the hit counts of each observed variant. 

        """     
        self.get_align_filename('lib_align.sam')
        print(f'Aligning reads against library: {self.lib_fasta}')
        self.minimap_align()
        self.analyze_align()
        self.count_hits()
        return

    def plot_hits(self, hitinds=[]):
        """"

        Plots all of the reads length vs. mean q-score and indicates hit reads.

        Usage:
        lvc.count_mapped_variants(hitinds=None)

        Parameters:
        -----------------------------------------------------
        hitinds, optional list of read indices to highlight in scatterplot
        Default behavior is just to plot all points that hit any variant.

        """
        if not hasattr(self, 'hits_df'):
            raise ValueError("Must run count_mapped_variants() first.")
            
        # check if fastqinfo exists. if not, create it.
        if not hasattr(self, 'fastqinfo'):
            self.fastq_info(makeplot=False)
            
        fqinfo_hits = self.fastqinfo.assign(in_hits = [False]*len(self.fastqinfo))
        if len(hitinds) == 0:
            print("Displaying all hits")
            fqinfo_hits.loc[self.align_df.index,'in_hits'] = True
        else:
            print("Displaying user-defined indices.")
            fqinfo_hits.loc[hitinds,'in_hits'] = True
            
        sns.scatterplot(data=fqinfo_hits, x='length', y='mean_q', hue='in_hits', s= 10)
        plt.show()
        return

    def write_hit_fastqs(self, variant, number, outname0):
        """"

        Writes some fastqs to file that were mapped to a particular variant.

        Usage:
        lvc.write_hit_fastq(variant, number, outname0)

        Parameters:
        -----------------------------------------------------
        variant: str, name of variant to extract
        number: int, number of requested reads to include in fastq
        outname0: str, name of output file (to be saved in reads_fastq directory)

        """ 
        if not hasattr(self, 'align_df'):
            raise ValueError("Must run count_mapped_variants() first.")
        
        # check if fastqinfo exists. if not, create it.
        if not hasattr(self, 'fastqinfo'):
            self.fastq_info(makeplot=False)
        
        if Path(outname0).parent != self.workdir:
            outname = self.workdir / (Path(outname0).stem + Path(outname0).suffix)
        else:
            outname = Path(outname0)
        if (outname.suffix != 'fastq') | (outname.suffix != 'fq'):
            outname = outname.parent / (outname.stem + '.fastq')
        
        var_df = self.align_df[self.align_df['ref_name'] == variant]
        num_vars = len(var_df)
        if num_vars == 0:
            print(f'{variant} not present.')
            print('Exiting with no action.')
            return
        elif num_vars < number:
            print(f'Only {num_vars} reads with {variant} present.')
            print('Writing all of them.')
            varinds = var_df.index
        else:
            varinds = self.fastqinfo.loc[var_df.index].sort_values(by='mean_q', ascending=False)[:number].index
        self.pop_fast(varinds, outname)

        return
class BarcodeDemultiplex():
    """"

    Class for demultiplexing barcoded amplicons and focusing on results at positions of interest.

    Usage:
        bd = BarcodeDemultiplex(barcode_csv, reads_fastq, amplicon_fasta)
        bd.fastq_info() (optional)
        bd.map_barcodes(min_overlap = 15, max_error = 0.1, num_cores = 8, makechart = True)
        bd.identify_edits(edit_position)
        bd.pop_fast(fast_ids, outname0, keepinds = True)
    """

    def __init__(self, barcode_csv, reads_fastq, amplicon_fasta):
        self.barcode_csv = Path(barcode_csv)
        self.reads_fastq = Path(reads_fastq)
        self.amplicon_fasta = Path(amplicon_fasta)

        print(f'Indexing fastq: "{str(self.reads_fastq)}"')
        self.fastq_index = SeqIO.index(str(self.reads_fastq), 'fastq')
        self.read_count = len(list(self.fastq_index.keys()))        
        print(f'Fastq contains {self.read_count} reads.')

        self.df_barcode = pd.read_csv(barcode_csv, header=None)
        col_num = len(self.df_barcode.columns)
        if col_num == 1:
            print('Assuming just one column with barcode sequences. Variant naming will be generic, ie. var_0, var_1, etc.')
            self.df_barcode.rename(columns={0:'barcode'}, inplace=True)
            self.df_barcode.insert(0, 'var_name', [f'var_{i}' for i in self.df_barcode.index])
        elif col_num == 2:
            print('Assuming two columns with variants names and associated barcode sequences.')
            self.df_barcode.rename(columns={0:'var_name', 1:'barcode'}, inplace=True)
        else:
            raise ValueError(f"Seeing {col_num} columns in barcode CSV and I don't know what to do.")
        self.df_barcode.head(10)
        self.col_num = col_num
        self.barcodes_fasta = self.barcode_csv.parent / ('barcodes.fasta')
        print(f'Building fasta of barcodes at {self.barcodes_fasta}')
        
        seqrecords = []
        for bcid, bc in zip(self.df_barcode['var_name'], self.df_barcode['barcode']):
            seqrecord = SeqRecord.SeqRecord(
                seq = Seq.Seq(bc),
                id = bcid,
                name = bcid,
                description = '')
            seqrecords.append(seqrecord)
        SeqIO.write(seqrecords, self.barcodes_fasta, 'fasta')
        
    def fastq_info(self, makeplot=True):
        """"

        Get some info about the fastq dataset.

        Usage:
        fastq_info(makeplot=True)

        Description:
        This gets some information on the fastq read dataset and includes a visualization
        of read length and quality if makeplot=True

        Parameters
        -----------------------------------------------------
        makeplot: boolean, display scatterplot of length vs. mean q-score if True

        Returns
        -----------------------------------------------------
        None
        But it will build a dataframe of read length and average q-scores and a dictionary of
        summary statistics.

        """
        read_info = dict()
        for read in self.fastq_index.keys():
            record = self.fastq_index[read]
            read_info[read] = {'length': len(record.seq), 'mean_q': np.mean(record.letter_annotations['phred_quality'])}
        self.fastqinfo = pd.DataFrame.from_dict(read_info, orient='index')

        if makeplot:
            sns.jointplot(data=self.fastqinfo, x='length', y='mean_q', s=5)
            plt.show()

        self.fastq_summary = {'total reads': len(self.fastqinfo),
                            'avg. length': np.mean(self.fastqinfo['length']),
                            'avg. q-score': np.mean(self.fastqinfo['mean_q'])}
        
        return
        
    def map_barcodes(self, min_overlap = 15, max_error = 0.1, num_cores = 8, makechart = True):
        """"

        Identify barcodes within amplicon reads
        
        Requirements, cutadapt.
        Get with: conda install bioconda::cutadapt

        Usage:
        bd.map_barcodes(min_overlap = 15, max_error = 0.1, num_cores = 8, makechart = True)

        Description:
        This maps the barcode sequences given in barcode_csv to amplicons in the reads fastq

        Parameters
        -----------------------------------------------------
        min_overlap: int, minimum allowable overlap of barcodes
        max_error: (int or float), if float must be between 0 and 1 for error rate. If int, represents max allowed mismatches
        num_cores: int, number of cores for cutadapt run (don't make this more than your computer has!)
        makechart: boolean, whether to show a pie chart with the distribution of barcodes among the reads
        -----------------------------------------------------
        None
        But it will build two dataframes: bd.df_reads with the mapping between every read id and its identified barcode,
        and bd.df_hits with the total observation counts for each variant.

        """        
        
        self.amplicon_fastq = self.reads_fastq.parent / 'amplicon.fastq'

        cutadapt_call = (f"cutadapt -b file:{self.barcodes_fasta}  --action none --overlap {min_overlap} --rename '{{id}} {{adapter_name}}' "
                        f"--discard-untrimmed --revcomp -e {max_error} -j {num_cores} --report minimal -o {self.amplicon_fastq} {self.reads_fastq}")

        run_sys_command(cutadapt_call)

        read_barcodes = SeqIO.parse(self.amplicon_fastq, 'fastq')
        read_bar_dict = {}
        for rb in read_barcodes:
            read_bar_dict[rb.id] = {'variant_id':rb.description.split(' ')[1]}
        
        self.df_reads = pd.DataFrame.from_dict(read_bar_dict, orient='index')
        bchits = np.unique(self.df_reads['variant_id'], return_counts=True)
        if self.col_num == 1:
            sortbc = np.argsort([int(bc.split('_')[1]) for bc in bchits[0]])
            hits = dict(zip(bchits[0][sortbc], bchits[1][sortbc]))
        else:
            hits = dict(zip(bchits[0], bchits[1]))
        df_hits = pd.DataFrame.from_dict(hits, orient='index')
        df_hits.rename(columns={0:'counts'}, inplace=True)
        self.df_hits = df_hits
        print(f"{df_hits['counts'].sum()}/{self.read_count} ({df_hits['counts'].sum()/self.read_count:.1%}) mapped to barcodes from {self.barcodes_fasta}.")
        if makechart:        
            colormap = plt.get_cmap('tab20')    
            labels = list(df_hits.index) + ['unmapped']
            counts = list(df_hits['counts']) + [self.read_count - df_hits['counts'].sum()]
            colors = [colormap(i / (len(counts) - 1)) for i in range(len(counts) - 1)] + ['gray']
            plt.pie(counts, labels=labels, colors=colors)
            plt.show()
        return


    def identify_edits(self, edit_position):
        """"

        Identify edit frequency at designated edit position
        
        Requirements, minimap2.
        Get with: conda install bioconda::minimap2

        Usage:
        bd.identify_edits(edit_position)

        Description:
        This aligns the reads with barcodes to the amplicon and then compares the read
        results at the designated edit position.

        Parameters
        -----------------------------------------------------
        edit_position: int, location of the expected edit from the amplicon fasta (0-indexed) 
        -----------------------------------------------------
        
        Returns:
        -----------------------------------------------------
        edit_percent dataframe
        (but also stores it in the object as bd.edit_percent)

        """      
        
        
        if ~hasattr(self, 'amplicon_sam'):
            print(f"Aligning amplicon sequence ({self.amplicon_fasta}) to reads with barcode matches ({self.amplicon_fastq}).")
            self.amplicon_sam = self.amplicon_fastq.parent / ("amplicon_barcode_align.sam")
            minimap2_call = f"minimap2 -ax map-ont {self.amplicon_fasta} {self.amplicon_fastq} > {self.amplicon_sam}"
            run_sys_command(minimap2_call)
        else:
            print(f"Amplicon alignment already exists ({self.amplicon_sam}). Continuing analysis without re-aligning.")
        
        edit_dict = {}
        for var in self.df_barcode['var_name']:
            edit_dict[var] = {'A':[], 'T':[], 'G':[], 'C':[]}
        
        counter = np.zeros(4, dtype=int)
        with pysam.AlignmentFile(self.amplicon_sam, "r") as alignments:
            for alignment in alignments:
                if alignment.is_unmapped or alignment.is_secondary or alignment.is_supplementary:
                    counter[0] += 1
                    continue
                # make sure the read covers the variable region
                covers_variable_region = (alignment.reference_start < edit_position) and (alignment.reference_end > edit_position)
                if not covers_variable_region:
                    counter[1] += 1
                    continue
                varname = self.df_reads.loc[alignment.query_name, 'variant_id']
                try:
                    pairwise = np.array(alignment.get_aligned_pairs(matches_only=True))                 
                    editind = pairwise[np.where(pairwise[:,1] == edit_position)[0][0],0]
                except:
                    counter[2] += 1
                    continue
                alignbase = alignment.query_sequence[editind]
                alignqual = alignment.query_qualities[editind]
                edit_dict[varname][alignbase].append(alignqual)
                counter[3] += 1
        
        print(f"Read sequence data at edit position collected from {counter[3]}/{self.df_hits['counts'].sum()} ({counter[3]/self.df_hits['counts'].sum():.1%}) of reads with barcodes.")
              
        edit_summary = dict()
        for key in edit_dict.keys():
            nuckeys = edit_dict[key].keys()
            editcounts = np.zeros(len(nuckeys))
            for nind, nuc in enumerate(nuckeys):
                editcounts[nind] = np.sum(1-10**(-np.array(edit_dict[key][nuc])/10))
            edit_summary[key] = dict(zip(nuckeys,editcounts*100/np.sum(editcounts)))

        self.edit_percent = pd.DataFrame.from_dict(edit_summary, orient='index')
        print(f"Computed read results (%) for position {edit_position}:")
        self.edit_percent
        return self.edit_percent

    def pop_fast(self, fast_ids, outname0, keepinds = True):
        # exports fasta or fastq files from read ids
        # check if fastqinfo exists. if not, create it.
        if not hasattr(self, 'fastqinfo'):
            self.fastq_info(makeplot=False)
        
        if Path(outname0).parent != self.workdir:
            outname = self.workdir / (Path(outname0).stem + Path(outname0).suffix)
        else:
            outname = outname0
    
        
        seqrecords = []
        for ind, fid in enumerate(fast_ids):
            try:
                seqrecord = self.fastq_index[fid]
                if keepinds:
                    nameind = fid
                else:
                    nameind = f'pop{ind}'
                    
                seqrecord.id = nameind
                seqrecord.name = nameind
                seqrecord.description = ''
                
                seqrecords.append(seqrecord)
            except:
                print(f'{fid} not found in index.')
        if (Path(outname).suffix == '.fa') | (Path(outname).suffix == '.fasta'):
            filetype = 'fasta'
        elif (Path(outname).suffix == '.fq') | (Path(outname).suffix == '.fastq'):
            filetype = 'fastq'
        else:
            raise ValueError(f'Unknown extension for {outname}. Must be fasta or fastq.') 
        print(f"Writing {len(seqrecords)} as {filetype} to {outname}")
        SeqIO.write(seqrecords, outname, filetype)
        return


        



        
        







    

    




