# Standard imports
import os
import re
import shutil

# External imports
import pandas as pd

# Local imports
from .constant import generate_tsm_audio_constant, modify_annots_tsm_constant
from .random import generate_tsm_audio_random, modify_annots_tsm_random
from .continuous import generate_tsm_audio_continuous
from .utils import get_query_timestamps, extract_audio_excerpt, modify_annots_select, get_audio_files

class QueryGenerator:
    """
    This class is used to generate time-scale modified audio queries and annotation files.
    It is used to generate time-scale modified audio queries and annotation files for the piano concerto accompaniment benchmark.
    
    Inputs
    audio_summary_file: the file that contains the audio summary
    query_measures_file: the file that contains the query measures
    audio_root: the root directory of the audio files
    annot_root: the root directory of the annotation files
    
    Outputs
    None
    """
    def __init__(self, audio_summary_file: str, query_measures_file: str, audio_root: str, annot_root: str):
        """
        Initializes the QueryGenerator class.
        
        Inputs
        audio_summary_file: the file that contains the audio summary
        query_measures_file: the file that contains the query measures
        audio_root: the root directory of the audio files
        annot_root: the root directory of the annotation files
        """
        self.audio_summary_file = audio_summary_file
        self.query_measures_file = query_measures_file
        self.audio_root = audio_root
        self.annot_root = annot_root

    def generateQueriesConstant(self, outdir, tsm_factors):
        '''
        Preps and generates time-scale modified audio queries and annotation files.
        
        Inputs
        outdir: directory to create and populate with audio queries
        tsm_factors: list of time-scale modification factors to use in generating queries
        '''
        
        if not os.path.exists(outdir):
            os.mkdir(outdir)    
        
        for p_file in get_audio_files(self.audio_summary_file, r'_P\d+.\S+$'): # all solo piano 
            
            base_id = os.path.splitext(p_file)[0] # e.g. rach2_mov1_P1
            piece_dir = f'{outdir}/{base_id}'
            
            if os.path.exists(piece_dir):
                print(f'Directory {piece_dir} already exists.  Skipping.')
                continue
            os.mkdir(piece_dir)
            
            for tsm_factor in tsm_factors:
                
                tsm_dir = f'{piece_dir}/tsm{tsm_factor:.2f}' # e.g. outdir/rach2_mov1_P1/tsm0.85
                os.mkdir(tsm_dir)
                
                # generate time-scale modified audio
                tsm_id = f'{base_id}_tsm{tsm_factor:.2f}_all' # e.g. rach2_mov1_P1_tsm0.85_all
                orig_audio_file = f'{self.audio_root}/{p_file}'
                tsm_audio_file = f'{tsm_dir}/{tsm_id}.wav'
                generate_tsm_audio_constant(orig_audio_file, tsm_audio_file, tsm_factor)
                
                # generate time-scale modified annotation file
                orig_annot_file = f'{self.annot_root}/{base_id}.beats'
                tsm_annot_file = f'{tsm_dir}/{tsm_id}.beats'
                modify_annots_tsm_constant(orig_annot_file, tsm_annot_file, tsm_factor)
                
                # get query start & end timestamps
                piece_id = re.sub(r'_P1$','', base_id) # e.g. rach2_mov1
                _, query_tuples = get_query_timestamps(piece_id, self.query_measures_file, tsm_annot_file)
                
                for cnt, (query_start, query_end) in enumerate(query_tuples):
                    
                    # generate query audio file
                    query_id = f'{base_id}_tsm{tsm_factor:.2f}_q{cnt+1}' # e.g. rach2_mov1_P1_tsm0.85_q1
                    query_audio_file = f'{tsm_dir}/{query_id}.wav'
                    extract_audio_excerpt(tsm_audio_file, query_audio_file, query_start, query_end)
                    
                    # generate query annotation file
                    query_annot_file = f'{tsm_dir}/{query_id}.beats'
                    modify_annots_select(tsm_annot_file, query_annot_file, query_start, query_end)
                    
    def generateQueriesRandom(self, outdir, max_tsm_factor: float, num_queries: int):
        """
        Generates time-scale modified audio queries and annotation files with random time-scale modification factors.
        
        Inputs
        outdir: directory to create and populate with audio queries
        max_tsm_factor: the maximum time-scale modification factor to use in generating queries
        num_queries: the number of queries to generate
        """
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            
        for p_file in get_audio_files(self.audio_summary_file, r'_P\d+.\S+$'): # all solo piano 
            base_id = os.path.splitext(p_file)[0] # e.g. rach2_mov1_P1
            piece_dir = f'{outdir}/{base_id}'
            
            tsm_dir = f'{piece_dir}/tsm_random_max{max_tsm_factor}' # e.g. outdir/rach2_mov1_P1/tsm_random_max2.00
            if os.path.exists(tsm_dir):
                # remove the directory
                shutil.rmtree(tsm_dir)
            os.mkdir(tsm_dir)
            
            for i in range(num_queries):
                
                # generate time-scale modified audio
                tsm_id = f'{base_id}_tsm_random_seed{i}' # e.g. rach2_mov1_P1_tsm_random_seed1
                orig_audio_file = f'{self.audio_root}/{p_file}'
                tsm_audio_file = f'{tsm_dir}/{tsm_id}.wav'
                tsm_alignment = generate_tsm_audio_random(orig_audio_file, tsm_audio_file, max_tsm_factor, seed=i)
                
                # generate time-scale modified annotation file
                orig_annot_file = f'{self.annot_root}/{base_id}.beats'
                tsm_annot_file = f'{tsm_dir}/{tsm_id}.beats'
                modify_annots_tsm_random(orig_annot_file, tsm_annot_file, tsm_alignment)
                
                # get query start & end timestamps
                piece_id = re.sub(r'_P1$','', base_id) # e.g. rach2_mov1
                _, query_tuples = get_query_timestamps(piece_id, self.query_measures_file, tsm_annot_file)
                for cnt, (query_start, query_end) in enumerate(query_tuples):
                    
                    # generate query audio file
                    query_id = f'{base_id}_tsm_random_seed{i}_q{cnt+1}' # e.g. rach2_mov1_P1_tsm_random_seed1_q1
                    query_audio_file = f'{tsm_dir}/{query_id}.wav'
                    extract_audio_excerpt(tsm_audio_file, query_audio_file, query_start, query_end)
                    
                    # generate query annotation file
                    query_annot_file = f'{tsm_dir}/{query_id}.beats'
                    modify_annots_select(tsm_annot_file, query_annot_file, query_start, query_end)
                    
    def generateQueriesContinuous(self, outdir, max_tsm_factor: float, num_queries: int):
        """
        Generates time-scale modified audio queries and annotation files with random time-scale modification factors.
        
        Inputs
        outdir: directory to create and populate with audio queries
        max_tsm_factor: the maximum time-scale modification factor to use in generating queries
        num_queries: the number of queries to generate
        """
        if not os.path.exists(outdir):
            os.mkdir(outdir)
            
        for p_file in get_audio_files(self.audio_summary_file, r'_P\d+.\S+$'): # all solo piano 
            base_id = os.path.splitext(p_file)[0] # e.g. rach2_mov1_P1
            piece_dir = f'{outdir}/{base_id}'
            
            tsm_dir = f'{piece_dir}/tsm_continuous_max{max_tsm_factor}' # e.g. outdir/rach2_mov1_P1/tsm_continuous_max2.00
            if os.path.exists(tsm_dir):
                # remove the directory
                shutil.rmtree(tsm_dir)
            os.mkdir(tsm_dir)
            
            for i in range(num_queries):
                
                # generate time-scale modified audio
                tsm_id = f'{base_id}_tsm_continuous_seed{i}' # e.g. rach2_mov1_P1_tsm_continuous_seed1
                orig_audio_file = f'{self.audio_root}/{p_file}'
                tsm_audio_file = f'{tsm_dir}/{tsm_id}.wav'
                tsm_alignment = generate_tsm_audio_continuous(orig_audio_file, tsm_audio_file, max_tsm_factor, seed=i)
                
                # generate time-scale modified annotation file
                orig_annot_file = f'{self.annot_root}/{base_id}.beats'
                tsm_annot_file = f'{tsm_dir}/{tsm_id}.beats'
                modify_annots_tsm_random(orig_annot_file, tsm_annot_file, tsm_alignment)
                
                # get query start & end timestamps
                piece_id = re.sub(r'_P1$','', base_id) # e.g. rach2_mov1
                _, query_tuples = get_query_timestamps(piece_id, self.query_measures_file, tsm_annot_file)
                for cnt, (query_start, query_end) in enumerate(query_tuples):
                    
                    # generate query audio file
                    query_id = f'{base_id}_tsm_continuous_seed{i}_q{cnt+1}' # e.g. rach2_mov1_P1_tsm_continuous_seed1_q1
                    query_audio_file = f'{tsm_dir}/{query_id}.wav'
                    extract_audio_excerpt(tsm_audio_file, query_audio_file, query_start, query_end)
                    
                    # generate query annotation file
                    query_annot_file = f'{tsm_dir}/{query_id}.beats'
                    modify_annots_select(tsm_annot_file, query_annot_file, query_start, query_end)