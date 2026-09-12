"""
MatchMaker OLTW baselines (Dixon 2005 and Arzt & Widmer 2010) via the pymatchmaker library.

Requirements:
- A conda env named `matchmaker` with pymatchmaker installed, or $MATCHMAKER_PYTHON
  pointing at its python interpreter.

Summary:
- Online processing: Run a MatchMaker frame-level OLTW follower on the precomputed piano
  query and reference features, and save the predicted alignment as hyp.npy.

MatchMaker's followers are score followers, but their position axis is just an array of
floats. The worker sets it to reference timestamps so they report reference seconds.

This module is not named matchmaker.py because matchmaker_worker.py sits beside it and
would then import this file instead of the installed package.
"""

import os
import logging
import subprocess

logger = logging.getLogger(__name__)

WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matchmaker_worker.py')
DEFAULT_ENV_NAME = 'matchmaker'


def verify_matchmaker_installation(python_path=None):
    """
    Verify that the matchmaker conda environment is available.

    Args:
        python_path: Path to the python interpreter of the matchmaker env. If None, uses
            $MATCHMAKER_PYTHON, then a sibling env of the active conda env.

    Returns:
        Path to the python interpreter.

    Raises:
        RuntimeError: If the environment or the worker script cannot be found.
    """
    if python_path is None:
        python_path = os.environ.get('MATCHMAKER_PYTHON')

    if python_path is None:
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            candidate = os.path.join(os.path.dirname(conda_prefix), DEFAULT_ENV_NAME, 'bin', 'python')
            if os.path.exists(candidate):
                python_path = candidate

    if python_path is None or not os.path.exists(python_path):
        raise RuntimeError(
            f'Could not locate the {DEFAULT_ENV_NAME} environment. '
            'Set MATCHMAKER_PYTHON to its python interpreter.'
        )
    if not os.path.exists(WORKER_PATH):
        raise RuntimeError(f'Worker script not found at {WORKER_PATH}')

    return python_path


def online_processing(scenario_dir, out_dir, p_ref_cache_dir, ref_start_time, method,
                      hop_length, sr, window_size=10.0, distance_metric='cosine',
                      step_size=None, python_path=None):
    """
    Online processing for a MatchMaker OLTW baseline.

    Runs the follower in the matchmaker conda env, which pins numpy<2 and so cannot share
    a process with this one. Saves the predicted alignment as hyp.npy (2 x N, in seconds,
    reference times relative to the full reference audio).

    Args:
        scenario_dir: Path to the scenario directory.
        out_dir: Path to the output directory (will be created).
        p_ref_cache_dir: Path to the reference feature .npy file.
        ref_start_time: Start time of the reference in seconds.
        method: 'dixon' or 'arzt'.
        hop_length: Hop length in samples.
        sr: Sample rate.
        window_size: Search window in seconds.
        distance_metric: 'cosine' or 'euclidean'.
        step_size: Max reference frames advanced per query frame (arzt only).
        python_path: Path to the matchmaker env python interpreter.

    Raises:
        RuntimeError: If the environment is missing or the worker fails.
    """
    python_cmd = verify_matchmaker_installation(python_path)

    # exist_ok, and cleaned up on failure: the worker is a subprocess with many ways to
    # fail, and a leftover empty directory would trip the caller's assert on every retry
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        python_cmd, WORKER_PATH,
        '--ref-feat', p_ref_cache_dir,
        '--query-feat', os.path.join(scenario_dir, 'pquery_stft.npy'),
        '--out', os.path.join(out_dir, 'hyp.npy'),
        '--method', method,
        '--sr', str(sr),
        '--hop-length', str(hop_length),
        '--window-size', str(window_size),
        '--distance-metric', distance_metric,
        '--ref-start-sec', str(ref_start_time),
    ]
    if step_size is not None:
        cmd += ['--step-size', str(step_size)]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if not os.listdir(out_dir):
            os.rmdir(out_dir)
        raise RuntimeError(f'MatchMaker worker failed for {scenario_dir}:\n{result.stderr}')

    logger.debug(f'MatchMaker {method} online: saved hyp.npy to {out_dir}')
