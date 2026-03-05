import os
from noa import compute_cosine_distance
import librosa as lb
from numba import jit, prange
import numpy as np
import system_utils
from hmc_mir.align import dtw

def verify_cache_dir(indir):
    '''
    Verifies that the specified cache directory has the required files.
    
    Inputs
    indir: The cache directory to verify (features/{piece_id})
    '''
    assert os.path.exists(f'{indir}/pref_stft.npy'), f'pref_stft.npy missing from {indir}'


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

def run_dtw(scenario_path, out_dir, p_ref_cache_dir, hop_length, sr, steps, weights):
    '''
    Carries out the 'online' processing for a simple offline subseq DTW system.
    Inputs:
    scenario_path -- str, path to a single scenario file
    out_dir -- str, directory to where the .hyp will be saved
    p_ref_cache_dir -- str, path to where the features are stored
    '''
    system_utils.verify_scenario_dir(scenario_path)
    # verify_cache_dir(p_ref_cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)

    #load query feautes
    #scenario_path assumes -> scenarios/{benchmark}/{mode}/{s_id}/
    pquery_feat_path = os.path.join(scenario_path, "pquery_stft.npy")
    F_pquery = np.load(pquery_feat_path) 
    #p_ref_cache_dir assumes -> features/{piece_id}/{piece_id}.features.npy
    F_pref = np.load(p_ref_cache_dir) 

    C = cosine_dist(F_pquery, F_pref)
    _, _, wp_AB = dtw.dtw(C, steps, weights, subseq=True)

    hop_sec = hop_length / sr
    np.save(f'{out_dir}/hyp.npy', wp_AB*hop_sec)

    return

#TO VERIFY IF IT CORRESPONDS TO OUR PREVIOUS RESULTS
# (it did for me -- Sayema)
out_dir = "experiments/train/constant/DTW/a/hyp.npy"
scenario_path = "scenarios/train/random/s1"
p_ref_cache_dir = "features/rach2_mov1_P1/chroma_stft_norm2/rach2_mov1_P1.features.npy"
hop_length = 512
sr = 22050
dtw_steps = np.array([[1,1],[1,2],[2,1]])
dtw_weights = np.array([1,1,2])
run_dtw(scenario_path, out_dir, p_ref_cache_dir, hop_length, sr, dtw_steps, dtw_weights)