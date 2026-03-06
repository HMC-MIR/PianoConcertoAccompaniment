"""
OLTW (Online Time Warping) system using PerformanceMatcher.jar.

Extracted from System_OLTW.ipynb.

Requirements:
- Java runtime environment (JRE) >= 21, accessible from command line.
- match/PerformanceMatcher.jar must be present.

Summary:
- Offline processing: Chop the piano reference (pref.wav) to the query region of interest
  and save as pref_chopped.wav in the scenario directory.
- Online processing: Run PerformanceMatcher.jar to align the piano query (p.wav) against
  the chopped piano reference, parse the output, convert to seconds, and save as hyp.npy.
"""

import os
import re
import logging
import subprocess
from shutil import which

import numpy as np
import librosa as lb
from scipy.io.wavfile import write

import system_utils

logger = logging.getLogger(__name__)

DEFAULT_JAR_PATH = 'match/PerformanceMatcher.jar'


def verify_oltw_installation(jar_path):
    """
    Verify that Java (>= 21) and PerformanceMatcher.jar are available.

    Args:
        jar_path: Path to PerformanceMatcher.jar.

    Returns:
        Path to the java executable.

    Raises:
        RuntimeError: If Java is not installed, is too old, or the jar is missing.
    """
    # Prefer conda Java if available
    java_cmd = None
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_java = os.path.join(conda_prefix, 'bin', 'java')
        if os.path.exists(conda_java):
            java_cmd = conda_java

    if java_cmd is None:
        java_cmd = which('java')

    if java_cmd is None:
        raise RuntimeError(
            'Java is not installed or not in PATH. '
            'Please install Java runtime environment (>= 21) to use OLTW.'
        )

    if not os.path.exists(jar_path):
        raise FileNotFoundError(
            f'PerformanceMatcher.jar not found at {jar_path}. '
            'Please ensure the JAR file is in the match/ directory.'
        )

    # Check Java version
    try:
        result = subprocess.run(
            [java_cmd, '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            text=True,
        )
        version_output = result.stdout
        version_match = re.search(r'version "(\d+)\.', version_output)
        if version_match:
            java_version = int(version_match.group(1))
            if java_version < 21:
                raise RuntimeError(
                    f'Java version {java_version} is too old. '
                    f'PerformanceMatcher.jar requires Java >= 21. '
                    f'Current Java: {java_cmd}\nVersion output: {version_output}'
                )
        else:
            logger.warning(f'Could not parse Java version. Output: {version_output}')
    except subprocess.TimeoutExpired:
        pass
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f'Could not verify Java version: {e}')

    return java_cmd


def offline_processing(scenario_dir, cache_dir, hop_length):
    """
    Offline processing for the OLTW system.

    Chops the piano reference audio (pref.wav) to the query region of interest
    and saves it as pref_chopped.wav in the scenario directory.

    Args:
        scenario_dir: Path to the scenario directory.
        cache_dir: Path to the cache directory (not used for OLTW, kept for compatibility).
        hop_length: Hop length in samples (not used for OLTW, kept for compatibility).
    """
    system_utils.verify_scenario_dir(scenario_dir)
    os.makedirs(cache_dir, exist_ok=True)

    out_path = os.path.join(scenario_dir, 'pref_chopped.wav')
    if os.path.exists(out_path):
        logger.debug(f'OLTW offline: {out_path} already exists. Skipping.')
        return

    pref_path = os.path.join(scenario_dir, 'pref.wav')
    if not os.path.exists(pref_path):
        raise FileNotFoundError(f'pref.wav missing in {scenario_dir}')

    p_start_t, p_end_t = system_utils.get_piano_reference_boundaries(scenario_dir)

    y_ref, sr_ref = lb.load(pref_path, sr=None)
    y_chopped = y_ref[int(p_start_t * sr_ref): int(p_end_t * sr_ref)]
    write(out_path, sr_ref, (y_chopped * 32767).astype('int16'))
    logger.debug(f'OLTW offline: saved chopped reference to {out_path}')


def parse_oltw_alignment(infile):
    """
    Parse the OLTW alignment text output file.

    Returns a list of (query_frame, ref_frame) tuples (1-based query, 0-based ref).
    """
    alignment_data = []
    with open(infile, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        if line.startswith('ALIGNMENT'):
            parts = line.split(' ')
            if len(parts) >= 3:
                query_idx = int(parts[1].rstrip(','))
                ref_idx = int(parts[2])
                alignment_data.append((query_idx, ref_idx))
    return alignment_data


def online_processing(scenario_dir, out_dir, hop_length, jar_path=None):
    """
    Online processing for the OLTW system.

    Runs PerformanceMatcher.jar to align the piano query (p.wav) against the chopped
    piano reference (pref_chopped.wav). Saves the predicted alignment as hyp.npy
    (2 x N, in seconds, reference times relative to the full reference audio).

    Args:
        scenario_dir: Path to the scenario directory.
        out_dir: Path to the output directory (will be created).
        hop_length: Hop length in samples (used for frame-to-second conversion).
        jar_path: Path to PerformanceMatcher.jar. Defaults to match/PerformanceMatcher.jar.

    Raises:
        RuntimeError: If Java is not installed or the jar is missing (logged and re-raised).
        FileNotFoundError: If pref_chopped.wav is missing (run offline_processing first).
    """
    if jar_path is None:
        jar_path = DEFAULT_JAR_PATH

    # Verify Java + jar — raises RuntimeError/FileNotFoundError with descriptive messages
    java_cmd = verify_oltw_installation(jar_path)

    system_utils.verify_scenario_dir(scenario_dir)
    assert not os.path.exists(out_dir), f'Output directory {out_dir} already exists.'
    os.makedirs(out_dir)

    pref_chopped_path = os.path.join(scenario_dir, 'pref_chopped.wav')
    if not os.path.exists(pref_chopped_path):
        raise FileNotFoundError(
            f'pref_chopped.wav not found in {scenario_dir}. '
            'Please run offline_processing first.'
        )

    pref_start_sec, _ = system_utils.get_piano_reference_boundaries(scenario_dir)

    query_path = os.path.join(scenario_dir, 'p.wav')
    alignment_output_path = os.path.join(out_dir, 'oltw_alignment.txt')

    cmd = [
        java_cmd, '-jar', jar_path,
        '-b', '-q', '-G', '-D', '--use-chroma-map',
        query_path, pref_chopped_path,
    ]

    with open(alignment_output_path, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)

    if result.returncode != 0:
        logger.warning(
            f'OLTW returned non-zero exit code {result.returncode}. '
            f'Check {alignment_output_path} for details.'
        )

    alignment_data = parse_oltw_alignment(alignment_output_path)
    if len(alignment_data) == 0:
        raise RuntimeError(f'No alignment data found in {alignment_output_path}')

    # Convert frame indices to seconds (query: 1-based → 0-based; ref: 0-based + offset)
    hop_sec = hop_length / 22050.0
    alignment_seconds = []
    for query_frame, ref_frame in alignment_data:
        query_sec = (query_frame - 1) * hop_sec
        ref_sec = ref_frame * hop_sec + pref_start_sec
        alignment_seconds.append((query_sec, ref_sec))

    alignment_array = np.array(alignment_seconds).T  # shape: 2 x N
    np.save(os.path.join(out_dir, 'hyp.npy'), alignment_array)

    # Clean up intermediate alignment text file
    if os.path.exists(alignment_output_path):
        os.remove(alignment_output_path)

    logger.debug(f'OLTW online: saved hyp.npy to {out_dir}')
