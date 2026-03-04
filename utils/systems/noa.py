"""
NOA (Naive Online Alignment) system.

Extracted from System_NOA.ipynb.

Summary:
- Offline processing: Compute chroma STFT features for the piano reference (pref.wav)
  and save to cache_dir/pref_stft.npy.
- Online processing: Align the piano query (p.wav) against the piano reference using
  the NOA algorithm, then save the predicted alignment as hyp.npy.
  Set monotonic=True for NOA_MONOTONIC.
"""

import os
import logging

import numpy as np
import librosa as lb

import system_utils
from noa import alignNOA, compute_cosine_distance, compute_euclidean_distance

logger = logging.getLogger(__name__)


def offline_processing(scenario_dir, cache_dir, hop_length):
    """
    Offline processing for the NOA system.

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
        logger.debug(f'NOA offline: {save_path} already exists. Skipping.')
        return

    pref_file = os.path.join(scenario_dir, 'pref.wav')
    if not os.path.exists(pref_file):
        raise FileNotFoundError(f'pref.wav missing in {scenario_dir}')

    y_pref, sr = lb.load(pref_file)
    F_pref = lb.feature.chroma_stft(y=y_pref, sr=sr, hop_length=hop_length, center=False, norm=2)
    np.save(save_path, F_pref)
    logger.debug(f'NOA offline: saved features to {save_path}')


def verify_cache_dir(cache_dir):
    """Verify that the cache directory contains the required NOA features."""
    assert os.path.exists(os.path.join(cache_dir, 'pref_stft.npy')), \
        f'pref_stft.npy missing from {cache_dir}'


def online_processing(scenario_dir, out_dir, cache_dir, hop_length, monotonic=False):
    """
    Online processing for the NOA system.

    Aligns the piano query (p.wav) against the piano reference using the NOA algorithm.
    Saves the predicted alignment as hyp.npy (2 x N, in seconds).

    Args:
        scenario_dir: Path to the scenario directory.
        out_dir: Path to the output directory (will be created).
        cache_dir: Path to the cache directory (must contain pref_stft.npy).
        hop_length: Hop length in samples.
        monotonic: If True, enforce monotonic alignment (NOA_MONOTONIC mode).
    """
    system_utils.verify_scenario_dir(scenario_dir)
    verify_cache_dir(cache_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)

    p_file = os.path.join(scenario_dir, 'p.wav')
    y, sr = lb.core.load(p_file)
    F_p = lb.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, center=False)
    F_pref = np.load(os.path.join(cache_dir, 'pref_stft.npy'))

    # Read reference start time from scenario.info
    info_file = os.path.join(scenario_dir, 'scenario.info')
    with open(info_file, 'r') as f:
        info = f.read().split()
    ref_start_time = float(info[-4])  # prefStart field

    wp = alignNOA(
        F_p, F_pref,
        ref_start_time=ref_start_time,
        cost_metric=compute_cosine_distance,
        monotonic=monotonic,
    )
    np.save(os.path.join(out_dir, 'hyp.npy'), wp)
    logger.debug(f'NOA online: saved hyp.npy to {out_dir}')
