"""
DTW (Naive Pairwise DTW) alignment system.

Extracted from System_NaivePairwiseDTW.ipynb.

Summary:
- Offline processing: Compute chroma STFT features for the piano reference (pref.wav)
  and save to cache_dir/pref_stft.npy.
- Online processing: Align the piano query (p.wav) against the piano reference using
  offline DTW with chroma features, then save the predicted alignment as hyp.npy.
"""

import os
import logging

import numpy as np
import librosa as lb
from numba import jit, prange
from hmc_mir.align import dtw

import system_utils

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True)
def cosine_dist(F1, F2):
    """Pairwise cosine distance matrix between two feature matrices (D x N, D x M) -> N x M."""
    F1 = F1.T
    F2 = F2.T
    C = np.zeros((F1.shape[0], F2.shape[0]))
    for row in prange(F1.shape[0]):
        for col in prange(F2.shape[0]):
            C[row, col] = 1 - np.dot(F1[row], F2[col]) / (
                np.linalg.norm(F1[row]) * np.linalg.norm(F2[col]) + 1e-9
            )
    return C


def offline_processing(scenario_dir, cache_dir, hop_length):
    """
    Offline processing for the DTW system.

    Computes chroma STFT features for the piano reference recording (pref.wav)
    and saves them to cache_dir/pref_stft.npy.

    Args:
        scenario_dir: Path to the scenario directory.
        cache_dir: Path to the cache directory.
        hop_length: Hop length in samples for chroma feature computation.
    """
    system_utils.verify_scenario_dir(scenario_dir)

    os.makedirs(cache_dir, exist_ok=True)

    save_path = os.path.join(cache_dir, 'pref_stft.npy')
    if os.path.exists(save_path):
        logger.debug(f'DTW offline: {save_path} already exists. Skipping.')
        return

    pref_file = os.path.join(scenario_dir, 'pref.wav')
    if not os.path.exists(pref_file):
        raise FileNotFoundError(f'pref.wav missing in {scenario_dir}')

    y_pref, sr = lb.load(pref_file)
    F_pref = lb.feature.chroma_stft(y=y_pref, sr=sr, hop_length=hop_length, center=False, norm=2)
    np.save(save_path, F_pref)
    logger.debug(f'DTW offline: saved features to {save_path}')


def verify_cache_dir(cache_dir):
    """Verify that the cache directory contains the required DTW features."""
    assert os.path.exists(os.path.join(cache_dir, 'pref_stft.npy')), \
        f'pref_stft.npy missing from {cache_dir}'


def online_processing(scenario_dir, out_dir, cache_dir, hop_length, steps, weights):
    """
    Online processing for the DTW system.

    Aligns the piano query (p.wav) against the piano reference using offline DTW
    with chroma features. Saves the predicted alignment as hyp.npy (2 x N, in seconds).

    Args:
        scenario_dir: Path to the scenario directory.
        out_dir: Path to the output directory (will be created).
        cache_dir: Path to the cache directory (must contain pref_stft.npy).
        hop_length: Hop length in samples.
        steps: L x 2 array of allowable DTW transitions.
        weights: Length-L array of DTW transition weights.
    """
    system_utils.verify_scenario_dir(scenario_dir)
    verify_cache_dir(cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)

    p_file = os.path.join(scenario_dir, 'p.wav')
    y, sr = lb.core.load(p_file)
    F_p = lb.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, center=False, norm=2)
    F_pref = np.load(os.path.join(cache_dir, 'pref_stft.npy'))

    C = cosine_dist(F_p, F_pref)
    _, _, wp_AB = dtw.dtw(C, steps, weights, True)

    hop_sec = hop_length / sr
    np.save(os.path.join(out_dir, 'hyp.npy'), wp_AB * hop_sec)
    logger.debug(f'DTW online: saved hyp.npy to {out_dir}')
