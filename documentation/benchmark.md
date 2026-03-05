You're a software developer trying to write a benchmark script for our research project. The goal of the project is to align a live audio with a given track stored in the backend. However, in this particular set of experiments, we're performing simulated real time alignment between the audio sequences.

Here are the different stages of our benchmark. We will describe the purpose of each stage, the inputs, the outputs, and the source code where you will build our benchmark script upon. The purpose of the benchmark script is to run everything in our notebooks in a python file so we can run async with tmux in terminal.

Make sure that you create proper logging info for each benchmark run, stored in logs/. If any configuration needs to be stored, store them in a folder called configs/ with proper json formatting.

As we write the benchmark script, we hope to clean up our codebase so that it's easier to access for future users of this repository.

We should be able to run the script in the command line and parse inputs with argparse. We should be able to input the following parameters:
* benchmark type: train/test
* query generation mode
    * specific parameters for each query generation mode as specified in json files in configs/.
* alignment system
    * specific parameters for each alignment system as specified in json files in configs/.

We should be able to run everything in stages or all together in the script.

# Prerequisites

The benchmark depends on the following directories being set up before running:

* `cfg_files/`: Contains configuration files that drive the benchmark.
    * `train.list`: Lists the concerto movements used for training (e.g., `rach2_mov1`).
    * `test.list`: Lists the concerto movements used for testing. Currently empty as we plan to expand the dataset.
    * `AudioDataSummary.csv`: Contains information about each audio recording, including URLs, licenses, and timestamps.
* `annot/`: Contains annotation files.
    * `.beats` files: Specify the timestamps of measure downbeats. There is one `.beats` file per piano-only and orchestra-only recording. Piano annotation files are soft links to their corresponding orchestra files (since they are synchronized by design).
    * `query.measures`: Defines the measure numbers of each music segment (contiguous chunks of solo piano playing) that will serve as queries.
    * `eval.measures`: Indicates which measures will be evaluated (only sections where both orchestra and piano are active).
* `audio/`: Contains the audio recordings (P, O, PO). See `01_DataPrep.ipynb` for download and setup instructions.
* `configs/`: Contains JSON configuration files for system parameters.
    * `default_systems.json`: Default configurations for all alignment systems (DTW, NOA, NOA_MONOTONIC, OLTW, OLTW_GLOBAL).
    * `oltw_global_examples.json`: Example OLTW_GLOBAL configurations with different parameter settings (A–F). See `configs/README.md` for usage details.
    * The coding agent may modify these configs as needed.

# Stage 0: Select Benchmark to Run

We currently have two benchmarks we would like to test on. We will have two different benchmarks, one is train and the other is test. The user should be able to select which benchmark to run.

# Stage 1: Query Generation

The audios of interest in our experiments are the piano concertos, stored in `audio/`. Each file in `audio/` has a `piece_id` that has a corresponding composer, movement, and recording type: P, O, or PO (`{composer}_{movement}_{P/O/PO}`). We are only interested in files that have P only. We use these as our source for generating our queries.

The different query generations can be found in `utils/query/` where each Python file corresponds to a query generation mode.

We use a time-scale constant called alpha to measure the ratio of tempos between the original track and the generated query.

The different modes include:
* Constant (`constant.py`): a global alpha that is constant throughout the entire query.
    * Parameters: `tsm_factor` (global alpha)
    * Default values: `[0.8, 0.9, 1, 1.1, 1.25]`
* Segmented (although it's called "random" in the current codebase) (`random.py`): Each query has a random number of segments, where each segment has a random alpha.
    * Parameters: `max_tsm_factor`
    * Default value: `2`
* Continuous (`continuous.py`): A continuous alpha that changes throughout the entire query, where alpha changes at every frame of the audio.
    * Parameters: `max_alpha_change`
    * Default value: `1.005`

For each mode, multiple seeded queries are generated per configuration:
* **Train benchmark**: `num_queries = 5`, using seeds `0–4`.
* **Test benchmark**: `num_queries = 10`, using seeds `100–109`.

The queries are then stored in `queries/{train/test}/{piece_id}/` as an audio file and its beats track.

What is a beats track? It notes the start of every measure in the piece, and after the query generation, we should be able to see a change in the beat track that corresponds to the newly time-scaled query. This file is what we use to verify the generation is correct.

Please verify that the generated queries are correct. What to check:
* Audio file should not be empty or corrupted
* The length of the beats track should be the same as the original track
* Make sure that the number of generated queries is correct and that the seeds are correct
    * File name structure: `{piece_id}_tsm_{mode}_seed{seed_number}_q{query_number}.{wav/beats}`

The code for how to run this is in notebook `01_DataPrep.ipynb`, in the "Audio Queries" section. Each query generation mode is labelled. Make sure that in the benchmark script you're setting the appropriate constants for filenames and folder structures.

# Stage 2: Generate Alignment Scenarios

The code for generating scenarios is in notebook `01_DataPrep.ipynb`, under the "Alignment Scenarios" section.

The benchmark consists of a set of alignment scenarios saved in the `scenarios/` directory. A single alignment scenario is defined as a tuple of three recordings:
* **Piano query**: The user's audio input, to be processed in an online fashion.
* **Orchestra only recording**: The accompaniment that we would like to time-scale modify in order to match the user's playing.
* **Full mix recording**: This recording serves as an intermediary that allows us to align the piano and orchestra recordings.

The goal of an alignment scenario is to accurately estimate where we are in the orchestra recording in an online fashion.

The alignment scenarios are numbered sequentially (e.g., `s1/`, `s2/`, etc.), and each scenario has its own directory containing the following:
* `p.wav`: A soft link to the piano query recording.
* `o.wav`: A soft link to the orchestra only recording.
* `po.wav`: A soft link to the full mix recording.
* `p.beats`: A soft link to the piano query annotation file.
* `o.beats`: A soft link to the orchestra annotation file.
* `scenario.info`: Contains information about the recordings in the scenario.

A `scenarios.summary` file is also generated at the mode level (e.g., `scenarios/constant/scenarios.summary`). Each line has the following fields:
* scenario id (e.g., s1, s2, etc.)
* piano file path
* orchestra file path
* full mix file path
* measure start (starting measure index, 1-based)
* measure end (ending measure index, inclusive)
* piano start (timestamp in seconds in the original full piano recording where the query begins)
* piano end (timestamp in seconds in the original full piano recording where the query ends)
* orchestra start (ground truth timestamp in seconds in the orchestra recording corresponding to the beginning of the query)
* orchestra end (ground truth timestamp in seconds in the orchestra recording corresponding to the end of the query)

Scenarios are generated per query generation mode. The `generateScenarios()` function in `01_DataPrep.ipynb` creates subdirectories for each mode (`constant/`, `random/`, `continuous/`) under the scenarios root and populates them with the scenario directories.

The key source files for scenario generation are:
* `utils/query/scenario.py`: Contains `generateScenariosConstant()`, `generateScenariosRandom()`, and `generateScenariosContinuous()`.
* `utils/query/utils.py`: Contains helper functions like `get_query_timestamps()` and `get_audio_files()`.

# Stage 3: Run Alignment

* We have 5 systems of interest:
    * **DTW**: Standard dynamic time warping using librosa. The code is contained in `System_NaivePairwiseDTW.ipynb`. We have been calling it "naive pairwise DTW," but in our benchmark script we wish to finally rename it to DTW. This is actually an offline alignment system, but we want to establish a baseline with it.
    * **NOA**: Our novel algorithm called Naive Online Alignment. The code is contained in `System_NOA.ipynb`.
    * **NOA_monotonic**: NOA, but the alignment has to be monotonic. The code is also in `System_NOA.ipynb`, but with the `monotonic` parameter set to `True`.
    * **OLTW**: Online time warping using `PerformanceMatcher.jar` found in `match/`. The code to run it is contained in `System_OLTW.ipynb`.
    * **OLTW_GLOBAL**: Our custom implementation of OLTW. You can find the implementation in `OnlineAlignment/core/alignment/offline/oltw.py`. We haven't implemented this in a notebook yet, but you can find sample code for how to run it in the code segment below.

Your goal in this stage would be to convert the notebooks to Python files that could be easily run using a common interface so we don't have to import many separate files in the benchmark script.

We want to test all query generation modes, i.e., all queries within these query generation modes (stored in scenarios), against all alignment systems. The different query generation mode results should be stored in their proper folder names under `experiments/{train/test}/{mode_name}/{alignment_system_name}/`.

This is a two-step process (offline/online processing):
* Offline processing should mostly involve chopping audio, copying audio, and computing features for the alignments. Make sure to reuse data whenever possible.
* Online processing is the alignment itself.

You can find the specifics within the respective notebooks.

During experiment running, make sure to add proper logging information and proper error handling.

Note on feature computation: Most alignment systems (DTW, NOA, NOA_MONOTONIC, OLTW_GLOBAL) use **chroma STFT features** (`chroma_stft` with `norm=2`), which should be precomputed once and stored in `features/chroma_stft_norm2/`. These features can be reused across systems. However, the **OLTW** system uses its own features computed internally by `PerformanceMatcher.jar` and does not use precomputed features.

System-specific parameters are defined in JSON configuration files stored in `configs/`. The default configurations are in `configs/default_systems.json`. For OLTW_GLOBAL, additional parameter settings (A–F) are available in `configs/oltw_global_examples.json`. See `configs/README.md` for full details.

Here is a sample code file for setting up experiments with a common interface:
```python
import os
import logging
import subprocess

import numpy as np
from numba import jit, prange
from hmc_mir.align import dtw
from tqdm import tqdm
import vamp
import pandas as pd

from noa import alignNOA,alignNOA_no_norm, compute_cosine_distance, compute_euclidean_distance
from utils.oltw import online_processing
from OnlineAlignment.core.alignment import run_offline_oltw

@jit(nopython=True, parallel=True)
def cosine_dist(F1, F2):
    '''
    Calculates the pairwise cosine distance matrix between two features matrices.

    Inputs
    F1: the first feature matrix, shape D x N
    F2: the second feature matrix, shape D x M

    Returns a pairwise cost matrix C of shape N x M, where elements indicate cosine distance.
    '''
    F1 = F1.T
    F2 = F2.T
    C = np.zeros((F1.shape[0], F2.shape[0]))
    for row in prange(F1.shape[0]):
        for col in prange(F2.shape[0]):
            C[row, col] = 1 - np.dot(F1[row], F2[col]) / (np.linalg.norm(F1[row]) * np.linalg.norm(F2[col]) + 1e-9)
    return C

@jit(nopython=True, parallel=True)
def euclidean_dist(F1, F2):
    '''
    Calculates the pairwise Euclidean distance matrix between two features matrices.

    Inputs
    F1: the first feature matrix, shape D x N
    F2: the second feature matrix, shape D x M

    Returns a pairwise cost matrix C of shape N x M, where elements indicate Euclidean distance.
    '''
    F1 = F1.T  # Now shape (N, D)
    F2 = F2.T  # Now shape (M, D)
    C = np.zeros((F1.shape[0], F2.shape[0]))
    for row in prange(F1.shape[0]):
        for col in prange(F2.shape[0]):
            diff = F1[row] - F2[col]
            C[row, col] = np.sqrt(np.sum(diff * diff))
    return C

def parse_match_outfile(infile):
            '''
            Parses the MATCH csv output file specifying the estimated alignment.
            
            Inputs
            infile: filepath to the MATCH csv output file
            
            Returns a 2xN array indicating the estimated alignment in seconds.
            '''
            d = pd.read_csv(infile, header=None)
            return np.vstack((d.loc[:,1], d.loc[:,2]))

class ExperimentRunner:
    def __init__(self, exp_type, kwargs, logger=None):
        """
        exp_type: experiment to run. Currently accepts DTW, NOA, or MATCH
        kwargs: arguments needed to pass in for the experiment
        logger: optional logger instance
        """
        self.exp_type = exp_type
        self.kwargs = kwargs
        self.logger = logger
        
    def run(self, scenarios_dir, out_dir):
        """
        Runs experiments for the given scenario and stores results to output directory.
        
        Example: scenarios_dir = "scenarios/s1", out_dir = "experiments"
        """
        scenario_id = scenarios_dir.split("/")[-1] # e.g. s1
        out_path = f"{out_dir}/{self.exp_type}/{scenario_id}" # e.g. experiments/DTW/s1
        
        # check if out_path exists. if so, skip
        if os.path.exists(out_path):
            print(f"Skipping {out_path} because it already exists")
            return
        
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # run experiment
        if self.exp_type == "DTW":
            self.run_dtw(scenarios_dir, out_path)
        elif self.exp_type == "NOA" or self.exp_type == "NOA_MONOTONIC":
            self.run_noa(scenarios_dir, out_path)
        elif self.exp_type == "MATCH":
            self.run_match(scenarios_dir, out_path)
        elif self.exp_type == "OLTW":
            self.run_oltw(scenarios_dir, out_path)
        elif "OLTW_GLOBAL" in self.exp_type:
            self.run_oltw_global(scenarios_dir, out_path)
        else:
            raise ValueError(f"Invalid experiment type: {self.exp_type}")
            
    def run_batch(self, scenarios_root, out_dir):
        """
        Runs experiments for all scenarios under scenarios_root.
        """
        if not os.path.isdir(scenarios_root):
            raise ValueError(f"{scenarios_root} is not a directory")
        
        for scenario_dir in tqdm(os.listdir(scenarios_root)):
            scenario_path = os.path.join(scenarios_root, scenario_dir)
            if os.path.isdir(scenario_path):
                try:
                    self.run(scenario_path, out_dir)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error running experiment for {scenario_path}: {e}", exc_info=True)
                    else:
                        print(f"Error running experiment for {scenario_path}: {e}")
                    continue
                
    def load_feat(self, scenarios_dir):
        """
        Loads features for the given scenario.
        """
        # load query and reference
        with open(os.path.join(scenarios_dir, "pair.txt"), "r") as f:
            query, reference = f.read().split()
            
        # load features
        query_feat_path = f"{self.kwargs['feat_dir']}/{query}.npy"
        reference_feat_path = f"{self.kwargs['feat_dir']}/{reference}.npy"
        query_feat = np.load(query_feat_path)
        reference_feat = np.load(reference_feat_path)
        
        return query_feat, reference_feat
                
    def run_dtw(self, scenarios_dir, out_path):
        """
        Runs DTW experiment for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # load query and reference features
        query_feat, reference_feat = self.load_feat(scenarios_dir)
        
        # run DTW
        if self.kwargs['distance_metric'] == 'cosine':
            C = cosine_dist(query_feat, reference_feat)
        elif self.kwargs['distance_metric'] == 'euclidean':
            C = euclidean_dist(query_feat, reference_feat)
        else:
            raise ValueError(f"Invalid distance metric: {self.kwargs['distance_metric']}")
        _, _, wp = dtw.dtw(C, self.kwargs['steps'], self.kwargs['weights'], True)
        
        # store result
        hop_sec = self.kwargs['hop_length'] / self.kwargs['sr']
        wp_sec = wp * hop_sec
        np.save(os.path.join(out_path, "hyp.npy"), wp_sec)
        
        
    def run_noa(self, scenarios_dir, out_path, monotonic = False):
        """
        Runs NOA experiment for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # load query and reference features
        query_feat, reference_feat = self.load_feat(scenarios_dir)
        
        # get distance metric
        if self.kwargs['distance_metric'] == 'cosine':
            cost_metric = compute_cosine_distance
        elif self.kwargs['distance_metric'] == 'euclidean':
            cost_metric = compute_euclidean_distance
        else:
            raise ValueError(f"Invalid distance metric: {self.kwargs['distance_metric']}")
        
        # run NOA
        norm = self.kwargs['norm']
        monotonic = self.kwargs['monotonic']
        if norm:
            wp = alignNOA(query_feat, reference_feat, cost_metric = cost_metric, monotonic = monotonic) # already in seconds
        else:
            wp = alignNOA_no_norm(query_feat, reference_feat, cost_metric = cost_metric, monotonic = monotonic) # already in seconds
        
        # store result
        np.save(os.path.join(out_path, "hyp.npy"), wp)
        
    def run_oltw_global(self, scenarios_dir, out_path):
        """
        Runs global OLTW experiment for the given scenario and stores results to output path.
        """
        # generate out_path
        os.makedirs(out_path, exist_ok=True)
        
        # load query and reference features
        query_feat, reference_feat = self.load_feat(scenarios_dir)
        
        # get distance metric
        if self.kwargs['distance_metric'] == 'cosine':
            cost_metric = compute_cosine_distance
        elif self.kwargs['distance_metric'] == 'euclidean':
            cost_metric = compute_euclidean_distance
        else:
            raise ValueError(f"Invalid distance metric: {self.kwargs['distance_metric']}")
        
        # parse steps and weights for window and transition
        DTW_steps = self.kwargs['DTW_steps']
        window_steps = self.kwargs['window_steps']
        DTW_weights = self.kwargs['DTW_weights']

        # run NOA
        wp = run_offline_oltw(reference_feat, query_feat, c=self.kwargs['c'], DTW_steps=DTW_steps, window_steps=window_steps, DTW_weights=DTW_weights)
        
        # convert to seconds
        hop_sec = self.kwargs['hop_length'] / self.kwargs['sr']
        wp_sec = wp * hop_sec
        
        # store result
        np.save(os.path.join(out_path, "hyp.npy"), wp_sec)
        
    def run_match(self, scenarios_dir, out_path):
        """
        Runs MATCH experiment for the given scenario and stores results to output path.
        """
        # load query and reference
        with open(os.path.join(scenarios_dir, "pair.txt"), "r") as f:
            query, reference = f.read().split()
            
        # load audio files
        query_audio_path = f"{self.kwargs['audio_root']}/{query}.wav"
        reference_audio_path = f"{self.kwargs['audio_root']}/{reference}.wav"
        
        match_align_filepath = f'{out_path}/match_p_pref.out'
        with open(match_align_filepath, 'w') as f:
            subprocess.run(['sonic-annotator', '-d', 'vamp:match-vamp-plugin:match:b_a', '-m', query_audio_path, reference_audio_path, '-w', 'csv', '--csv-stdout'],
                           check=True, stdout=f, stderr=subprocess.DEVNULL)
            
        # store result
        wp= parse_match_outfile(match_align_filepath)
        np.save(os.path.join(out_path, "hyp.npy"), wp)
        
    def run_oltw(self, scenarios_dir, out_path):
        online_processing(scenarios_dir, out_path, self.kwargs['hop_length'])
```

# Stage 4: Evaluation

We want to evaluate all query generation modes, i.e., all queries within these query generation modes (stored in scenarios), against all alignment systems. The different query generation mode results should be stored in their proper folder names under `eval/{train/test}/{mode_name}/{alignment_system_name}/`.
* Make sure to add proper logging information and proper error handling.

All the details, including code, for evaluation are in the notebook `03_Evaluate.ipynb`.

# Sample Benchmark Script

To help you understand our benchmarking process, here is the sample benchmark code from another project. Your benchmark code should follow a similar structure and formatting, plus what is specified in this document.

```python
#!/usr/bin/env python3
"""
Benchmark Pipeline Script

This script provides a command-line interface for running the SimRealtimeMazurkaBenchmark pipeline.
It supports preparing scenarios, computing features, running experiments, and evaluating results.
"""

import os
import sys
import argparse
import json
import pickle
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import librosa as lb
from tqdm import tqdm

# Import local modules
import utils.constants as constants
from utils.experiments import ExperimentRunner
from utils.match_features import extract_match_features
import eval_tools


# ============================================================================
# Configuration and Constants
# ============================================================================

BENCHMARK_CONFIGS = {
    'train_small': {
        'train_file': 'cfg/mazurkas.train.pkl',
        'pair_file': 'cfg/mazurkas.train_pairs.pkl',
        'scenarios_dir': 'scenarios',
        'experiments_dir': 'experiments',
        'eval_dir': 'eval',
    },
    'train': {
        'train_file': 'cfg/mazurkas.train_large.pkl',
        'pair_file': 'cfg/mazurkas.train_pairs_large.pkl',
        'scenarios_dir': 'scenarios_train',
        'experiments_dir': 'experiments_train',
        'eval_dir': 'eval_train',
    },
    'test': {
        'train_file': 'cfg/mazurkas.test.pkl',
        'pair_file': 'cfg/mazurkas.test_pairs.pkl',
        'scenarios_dir': 'scenarios_test',
        'experiments_dir': 'experiments_test',
        'eval_dir': 'eval_test',
    },
}

AUDIO_ROOT = "Chopin_Mazurkas/wav_22050_mono/Chopin_Op017No4"
ANNOT_ROOT = "Chopin_Mazurkas/annotations_beat/Chopin_Op017No4"
FEAT_DIR = "features"


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"benchmark_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to {log_file}")
    return logger


# ============================================================================
# Data Preparation Functions
# ============================================================================

def validate_file(filepath: str, logger: logging.Logger) -> bool:
    """
    Validate that a file exists, is a regular file, and is not empty.
    
    Args:
        filepath: path to file
        logger: logger instance
        
    Returns:
        True if file is valid, False otherwise
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return False
        
    if not os.path.isfile(filepath):
        logger.warning(f"Not a file: {filepath}")
        return False
        
    if os.path.getsize(filepath) == 0:
        logger.warning(f"File is empty: {filepath}")
        return False
        
    return True


def generate_scenarios(outdir: str, pairs_list: List[tuple], logger: logging.Logger):
    """
    Generate all possible alignment scenarios for a given output directory.
    
    Args:
        outdir: output directory
        pairs_list: list of pairs of recordings to process. Each pair is a tuple of (query, reference)
        logger: logger instance
    """
    logger.info(f"Generating scenarios in {outdir}/")
    
    # Create output directory
    if os.path.exists(outdir):
        logger.warning(f"Directory {outdir}/ already exists. Deleting and regenerating...")
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    
    cwd = os.getcwd()
    valid_scenarios_count = 0
    
    for i, (query, ref) in enumerate(tqdm(pairs_list, desc="Creating scenarios")):
        # Define source paths
        query_path = os.path.join(cwd, AUDIO_ROOT, f"{query}.wav")
        ref_path = os.path.join(cwd, AUDIO_ROOT, f"{ref}.wav")
        query_annot_path = os.path.join(cwd, ANNOT_ROOT, f"{query}.beat")
        ref_annot_path = os.path.join(cwd, ANNOT_ROOT, f"{ref}.beat")
        
        # Validate all source files
        if not all([
            validate_file(query_path, logger),
            validate_file(ref_path, logger),
            validate_file(query_annot_path, logger),
            validate_file(ref_annot_path, logger)
        ]):
            logger.warning(f"Skipping scenario {query} vs {ref} due to invalid files")
            continue

        scenario_id = f"s{valid_scenarios_count+1}"
        # create scenario directory
        scenario_dir = os.path.join(outdir, scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # generate symbolic links for query and reference audio files
        query_link = os.path.join(scenario_dir, "query.wav")
        ref_link = os.path.join(scenario_dir, "ref.wav")
        os.symlink(query_path, query_link)
        os.symlink(ref_path, ref_link)
        
        # generate symbolic links for annotation files
        query_annot_link = os.path.join(scenario_dir, "query.beats")
        ref_annot_link = os.path.join(scenario_dir, "ref.beats")
        os.symlink(query_annot_path, query_annot_link)
        os.symlink(ref_annot_path, ref_annot_link)
        
        # generate text file storing the query and reference
        text = f"{query} {ref}\n"
        with open(os.path.join(scenario_dir, "pair.txt"), "w") as f:
            f.write(text)
            
        valid_scenarios_count += 1
    
    logger.info(f"Generated {valid_scenarios_count} scenarios (skipped {len(pairs_list) - valid_scenarios_count})")


def compute_chroma_stft_features(piece_ids: List[str], logger: logging.Logger):
    """Compute and save chroma_stft features for given pieces."""
    chroma_stft_dir = f"{FEAT_DIR}/chroma_stft_norm2"
    os.makedirs(chroma_stft_dir, exist_ok=True)
    
    logger.info(f"Computing chroma_stft features for {len(piece_ids)} pieces")
    
    for piece_id in tqdm(piece_ids, desc="Computing chroma_stft"):
        feat_path = f"{chroma_stft_dir}/{piece_id}.npy"
        
        # Skip if already exists
        if os.path.exists(feat_path):
            logger.debug(f"Skipping {piece_id} - already computed")
            continue
        
        audio_path = f"{AUDIO_ROOT}/{piece_id}.wav"
        y, sr = lb.load(audio_path)
        chroma_stft_feat = lb.feature.chroma_stft(
            y=y, sr=sr, 
            hop_length=constants.DEFAULT_HOP_LENGTH, 
            center=False, 
            norm=2
        )
        np.save(feat_path, chroma_stft_feat)
    
    logger.info("Chroma STFT features computed")


def compute_match_features(piece_ids: List[str], logger: logging.Logger):
    """Compute and save match features for given pieces."""
    match_dir = f"{FEAT_DIR}/match"
    os.makedirs(match_dir, exist_ok=True)
    
    logger.info(f"Computing match features for {len(piece_ids)} pieces")
    
    for piece_id in tqdm(piece_ids, desc="Computing match features"):
        feat_path = f"{match_dir}/{piece_id}.npy"
        
        # Skip if already exists
        if os.path.exists(feat_path):
            logger.debug(f"Skipping {piece_id} - already computed")
            continue
        
        audio_path = f"{AUDIO_ROOT}/{piece_id}.wav"
        match_feat = extract_match_features(audio_path)
        np.save(feat_path, match_feat)
    
    logger.info("Match features computed")


# ============================================================================
# Configuration Management
# ============================================================================

def load_system_config(config_path: Optional[str], systems: List[str], logger: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """
    Load system configuration from JSON file or use defaults.
    
    Args:
        config_path: Path to JSON configuration file, or None for defaults
        systems: List of system names to configure
        logger: Logger instance
    
    Returns:
        Dictionary mapping system names to their configurations
    """
    if config_path:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    # Use default configurations
    logger.info("Using default system configurations")
    return get_default_configs(systems)


def get_default_configs(systems: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get default configurations for specified systems."""
    configs = {}
    
    for system in systems:
        if system == 'DTW':
            configs[system] = {
                "steps": constants.DEFAULT_DTW_STEPS.tolist(),
                "weights": constants.DEFAULT_DTW_WEIGHTS.tolist(),
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "distance_metric": "cosine"
            }
        elif system in ['NOA', 'NOA_MONOTONIC']:
            configs[system] = {
                "steps": constants.DEFAULT_DTW_STEPS.tolist(),
                "weights": constants.DEFAULT_DTW_WEIGHTS.tolist(),
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "norm": True,
                "distance_metric": "cosine",
                "monotonic": system == 'NOA_MONOTONIC'
            }
        elif system == 'MATCH':
            configs[system] = {
                "audio_root": AUDIO_ROOT
            }
        elif system == 'OLTW':
            configs[system] = {
                "hop_length": constants.DEFAULT_HOP_LENGTH
            }
        elif system.startswith('OLTW_GLOBAL'):
            # Default OLTW_GLOBAL (Setting A)
            configs[system] = {
                "hop_length": constants.DEFAULT_HOP_LENGTH,
                "feat_dir": f"{FEAT_DIR}/chroma_stft_norm2",
                "sr": constants.DEFAULT_SR,
                "distance_metric": "cosine",
                "c": None,  # set to global
                "DTW_steps": [[1, 0], [0, 1], [1, 1]],
                "DTW_weights": [1, 1, 1],
                "window_steps": [[1, 1], [1, 0], [0, 1]]
            }
    
    return configs


# ============================================================================
# Command Implementations
# ============================================================================

def cmd_prepare(args, logger: logging.Logger):
    """Prepare scenarios for the specified benchmark."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    
    # Load pair list
    pair_file = config['pair_file']
    if not os.path.exists(pair_file):
        logger.error(f"Pair file not found: {pair_file}")
        logger.error("Please run data preparation first to create training pairs")
        sys.exit(1)
    
    with open(pair_file, 'rb') as f:
        pairs_list = pickle.load(f)
    
    # Generate scenarios
    generate_scenarios(config['scenarios_dir'], pairs_list, logger)
    
    logger.info("Scenario preparation complete")


def cmd_features(args, logger: logging.Logger):
    """Compute features for the specified benchmark and systems."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    
    # Load piece IDs
    train_file = config['train_file']
    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        logger.error("Please run data preparation first to create training set")
        sys.exit(1)
    
    with open(train_file, 'rb') as f:
        piece_ids = pickle.load(f)
    
    logger.info(f"Processing {len(piece_ids)} pieces for {args.benchmark} benchmark")
    
    # Compute required features based on systems
    compute_chroma_stft_features(piece_ids, logger)
    
    logger.info("Feature computation complete")


def cmd_experiment(args, logger: logging.Logger):
    """Run experiments for the specified benchmark and systems."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    scenarios_dir = config['scenarios_dir']
    exp_dir = config['experiments_dir']
    
    # Check scenarios exist
    if not os.path.exists(scenarios_dir):
        logger.error(f"Scenarios directory not found: {scenarios_dir}")
        logger.error("Please run 'prepare' command first")
        sys.exit(1)
    
    # Load system configurations
    system_configs = load_system_config(args.config, args.systems, logger)
    
    # Clean and create experiment directories
    logger.info(f"Cleaning experiment directory: {exp_dir}")
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    
    for system in args.systems:
        os.makedirs(os.path.join(exp_dir, system), exist_ok=True)
    
    # Run experiments for each system
    for system in args.systems:
        logger.info(f"Running {system} experiments")
        
        if system not in system_configs:
            logger.error(f"No configuration found for system: {system}")
            continue
        
        # Convert lists back to numpy arrays for DTW steps/weights
        kwargs = system_configs[system].copy()
        if 'steps' in kwargs:
            kwargs['steps'] = np.array(kwargs['steps']).reshape((-1, 2))
        if 'weights' in kwargs:
            kwargs['weights'] = np.array(kwargs['weights'])
        if 'DTW_steps' in kwargs:
            kwargs['DTW_steps'] = np.array(kwargs['DTW_steps'])
        if 'window_steps' in kwargs:
            kwargs['window_steps'] = np.array(kwargs['window_steps'])
        
        runner = ExperimentRunner(system, kwargs, logger=logger)
        runner.run_batch(scenarios_dir, exp_dir)
        
        logger.info(f"{system} experiments complete")
    
    logger.info("All experiments complete")


def cmd_evaluate(args, logger: logging.Logger):
    """Evaluate experiments for the specified benchmark."""
    if args.benchmark == 'test':
        logger.error("Test benchmark not yet implemented")
        sys.exit(1)
    
    config = BENCHMARK_CONFIGS[args.benchmark]
    scenarios_dir = config['scenarios_dir']
    exp_dir = config['experiments_dir']
    eval_dir = config['eval_dir']
    
    # Check experiments exist
    if not os.path.exists(exp_dir):
        logger.error(f"Experiments directory not found: {exp_dir}")
        logger.error("Please run 'experiment' command first")
        sys.exit(1)
    
    # Find all system subdirectories
    systems = [d for d in os.listdir(exp_dir) 
               if os.path.isdir(os.path.join(exp_dir, d))]
    
    if not systems:
        logger.warning(f"No systems found in {exp_dir}")
        return
    
    logger.info(f"Evaluating {len(systems)} systems: {', '.join(systems)}")
    
    # Evaluate each system
    for system in systems:
        logger.info(f"Evaluating {system}")
        system_exp_dir = os.path.join(exp_dir, system)
        system_eval_dir = os.path.join(eval_dir, system)
        
        eval_tools.eval_alignment_batch(system_exp_dir, scenarios_dir, system_eval_dir, logger=logger)
        logger.info(f"{system} evaluation complete")
    
    logger.info("All evaluations complete")


def cmd_run(args, logger: logging.Logger):
    """Run the full pipeline: prepare -> features -> experiment -> evaluate."""
    logger.info("=" * 80)
    logger.info(f"Running full pipeline for benchmark: {args.benchmark}")
    logger.info(f"Systems: {', '.join(args.systems)}")
    logger.info("=" * 80)
    
    # Step 1: Prepare scenarios
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Preparing scenarios")
    logger.info("=" * 80)
    cmd_prepare(args, logger)
    
    # Step 2: Compute features
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Computing features")
    logger.info("=" * 80)
    cmd_features(args, logger)
    
    # Step 3: Run experiments
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Running experiments")
    logger.info("=" * 80)
    cmd_experiment(args, logger)
    
    # Step 4: Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Evaluating results")
    logger.info("=" * 80)
    cmd_evaluate(args, logger)
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SimRealtimeMazurkaBenchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default settings
  python benchmark.py run --benchmark train_small --systems DTW NOA
  
  # Run full pipeline with custom config
  python benchmark.py run --benchmark train_small --config configs/my_config.json
  
  # Run individual steps
  python benchmark.py prepare --benchmark train_small
  python benchmark.py features --benchmark train_small --systems DTW NOA
  python benchmark.py experiment --benchmark train_small --systems OLTW_GLOBAL --config configs/oltw_config.json
  python benchmark.py evaluate --benchmark train_small
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Prepare command
    prepare_parser = subparsers.add_parser('prepare', help='Generate scenarios')
    prepare_parser.add_argument('--benchmark', required=True, 
                                choices=['train_small', 'train', 'test'],
                                help='Benchmark to prepare')
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Compute features')
    features_parser.add_argument('--benchmark', required=True,
                                 choices=['train_small', 'train', 'test'],
                                 help='Benchmark to compute features for')
    
    # Experiment command
    experiment_parser = subparsers.add_parser('experiment', help='Run experiments')
    experiment_parser.add_argument('--benchmark', required=True,
                                   choices=['train_small', 'train', 'test'],
                                   help='Benchmark to run experiments on')
    experiment_parser.add_argument('--systems', nargs='+', required=True,
                                   help='Systems to run (DTW, NOA, NOA_MONOTONIC, MATCH, OLTW, OLTW_GLOBAL, or custom)')
    experiment_parser.add_argument('--config', type=str,
                                   help='JSON configuration file for system parameters')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate experiments')
    evaluate_parser.add_argument('--benchmark', required=True,
                                 choices=['train_small', 'train', 'test'],
                                 help='Benchmark to evaluate')
    
    # Run command (full pipeline)
    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument('--benchmark', required=True,
                            choices=['train_small', 'train', 'test'],
                            help='Benchmark to run')
    run_parser.add_argument('--systems', nargs='+', required=True,
                            help='Systems to run (DTW, NOA, NOA_MONOTONIC, MATCH, OLTW, OLTW_GLOBAL, or custom)')
    run_parser.add_argument('--config', type=str,
                            help='JSON configuration file for system parameters')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Execute command
        if args.command == 'prepare':
            cmd_prepare(args, logger)
        elif args.command == 'features':
            cmd_features(args, logger)
        elif args.command == 'experiment':
            cmd_experiment(args, logger)
        elif args.command == 'evaluate':
            cmd_evaluate(args, logger)
        elif args.command == 'run':
            cmd_run(args, logger)
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
```