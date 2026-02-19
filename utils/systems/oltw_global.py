"""
OLTW_GLOBAL system — custom offline OLTW implementation.

Uses OnlineAlignment.core.alignment.run_offline_oltw.

Summary:
- Offline processing: Compute chroma STFT features for the piano reference (pref.wav)
  and save to cache_dir/pref_stft.npy.
- Online processing: Run run_offline_oltw on the query and reference features,
  convert to seconds, and save as hyp.npy.
"""

import os
import logging

import numpy as np
import librosa as lb
from numba import jit, prange

import system_utils
from OnlineAlignment.core.alignment import run_offline_oltw

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
    Offline processing for the OLTW_GLOBAL system.

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
        logger.debug(f'OLTW_GLOBAL offline: {save_path} already exists. Skipping.')
        return

    pref_file = os.path.join(scenario_dir, 'pref.wav')
    if not os.path.exists(pref_file):
        raise FileNotFoundError(f'pref.wav missing in {scenario_dir}')

    y_pref, sr = lb.load(pref_file)
    F_pref = lb.feature.chroma_stft(y=y_pref, sr=sr, hop_length=hop_length, center=False, norm=2)
    np.save(save_path, F_pref)
    logger.debug(f'OLTW_GLOBAL offline: saved features to {save_path}')


def verify_cache_dir(cache_dir):
    """Verify that the cache directory contains the required OLTW_GLOBAL features."""
    assert os.path.exists(os.path.join(cache_dir, 'pref_stft.npy')), \
        f'pref_stft.npy missing from {cache_dir}'


def online_processing(
    scenario_dir, out_dir, cache_dir, hop_length,
    c, DTW_steps, DTW_weights, window_steps,
):
    """
    Online processing for the OLTW_GLOBAL system.

    Runs run_offline_oltw on the piano query vs. piano reference features.
    Saves the predicted alignment as hyp.npy (2 x N, in seconds).

    Args:
        scenario_dir: Path to the scenario directory.
        out_dir: Path to the output directory (will be created).
        cache_dir: Path to the cache directory (must contain pref_stft.npy).
        hop_length: Hop length in samples.
        c: Window size parameter for OLTW (None = global).
        DTW_steps: Array of DTW step sizes.
        DTW_weights: Array of DTW step weights.
        window_steps: Array of window transition steps.
    """
    system_utils.verify_scenario_dir(scenario_dir)
    verify_cache_dir(cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)

    p_file = os.path.join(scenario_dir, 'p.wav')
    y, sr = lb.core.load(p_file)
    F_p = lb.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, center=False, norm=2)
    F_pref = np.load(os.path.join(cache_dir, 'pref_stft.npy'))

    wp = run_offline_oltw(
        F_pref, F_p,
        c=c,
        DTW_steps=DTW_steps,
        window_steps=window_steps,
        DTW_weights=DTW_weights,
    )

    hop_sec = hop_length / sr
    wp_sec = wp * hop_sec
    np.save(os.path.join(out_dir, 'hyp.npy'), wp_sec)
    logger.debug(f'OLTW_GLOBAL online: saved hyp.npy to {out_dir}')
