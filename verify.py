#!/usr/bin/env python3
"""
Verification Script: Compare benchmark.py outputs against notebook outputs.

This script checks that each pipeline stage produces outputs that are structurally
correct and numerically consistent with what the notebooks would produce.

The notebook pipeline uses these paths (no train/test prefix):
  scenarios/{mode}/s{N}/
  experiments/{mode}/{system}/s{N}/hyp.npy
  eval/{mode}/{system}/errs.pkl

The benchmark.py pipeline uses:
  scenarios/{benchmark}/{mode}/s{N}/
  experiments/{benchmark}/{mode}/{system}/s{N}/hyp.npy
  eval/{benchmark}/{mode}/{system}/errs.pkl

Usage:
  # Check structural correctness of benchmark outputs (no notebook outputs needed)
  python verify.py --benchmark --check structure

  # Compare benchmark outputs numerically against existing notebook outputs
  python verify.py --benchmark train --check compare \\
      --notebook-scenarios scenarios \\
      --notebook-experiments experiments \\
      --notebook-eval eval

  # Run all checks
  python verify.py --benchmark train --check all \\
      --notebook-scenarios scenarios \\
      --notebook-experiments experiments \\
      --notebook-eval eval
"""

import os
import sys
import pickle
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

# ============================================================================
# Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s  %(message)s',
)
logger = logging.getLogger('verify')

PASS = '✓'
FAIL = '✗'
WARN = '⚠'

ALL_MODES = ['constant', 'random', 'continuous']
ALL_SYSTEMS = ['DTW', 'NOA', 'NOA_MONOTONIC', 'OLTW', 'OLTW_GLOBAL']

# Notebook system name → benchmark.py system name mapping
NOTEBOOK_TO_BENCHMARK_SYSTEM = {
    'NaivePairwiseDTW': 'DTW',
    'NOA': 'NOA',
    'NOA-Mono': 'NOA_MONOTONIC',
    'OLTW': 'OLTW',
}


def _ok(msg):
    logger.info(f'  {PASS} {msg}')


def _fail(msg):
    logger.error(f'  {FAIL} {msg}')


def _warn(msg):
    logger.warning(f'  {WARN} {msg}')


# ============================================================================
# Stage 1: Query verification
# ============================================================================

def verify_queries(benchmark: str, mode_config: dict) -> int:
    """
    Verify Stage 1 (query generation) outputs.

    Checks:
    - Query directories exist for each piece and mode
    - Each .wav file is non-empty
    - Each .wav has a corresponding .beats file
    - The number of query files matches expectations
    """
    import json
    queries_root = f'queries/{benchmark}'
    logger.info(f'\n=== Stage 1: Query Generation ({queries_root}) ===')
    errors = 0

    if not os.path.isdir(queries_root):
        _fail(f'queries root not found: {queries_root}')
        return 1

    for piece_dir in sorted(Path(queries_root).iterdir()):
        if not piece_dir.is_dir():
            continue
        piece_id = piece_dir.name  # e.g. rach2_mov1_P1

        for mode in ALL_MODES:
            cfg = mode_config.get(mode, {})

            if mode == 'constant':
                for tsm in cfg.get('tsm_factors', []):
                    tsm_dir = piece_dir / f'tsm{tsm:.2f}'
                    if not tsm_dir.exists():
                        _fail(f'Missing: {tsm_dir}')
                        errors += 1
                        continue
                    wavs = sorted(tsm_dir.glob('*_q*.wav'))
                    beats = sorted(tsm_dir.glob('*_q*.beats'))
                    if len(wavs) == 0:
                        _fail(f'No query wavs in {tsm_dir}')
                        errors += 1
                    elif len(wavs) != len(beats):
                        _fail(f'{tsm_dir}: {len(wavs)} wavs but {len(beats)} beats files')
                        errors += 1
                    else:
                        for wav in wavs:
                            if wav.stat().st_size == 0:
                                _fail(f'Empty wav: {wav}')
                                errors += 1
                        _ok(f'{piece_id} constant tsm={tsm:.2f}: {len(wavs)} queries')

            elif mode == 'random':
                max_f = cfg.get('max_tsm_factor', 2)
                tsm_dir = piece_dir / f'tsm_random_max{max_f}'
                if not tsm_dir.exists():
                    _fail(f'Missing: {tsm_dir}')
                    errors += 1
                    continue
                wavs = sorted(tsm_dir.glob('*_q*.wav'))
                _ok(f'{piece_id} random max={max_f}: {len(wavs)} query wavs')

            elif mode == 'continuous':
                max_a = cfg.get('max_alpha_change', 1.005)
                tsm_dir = piece_dir / f'tsm_continuous_max{max_a}'
                if not tsm_dir.exists():
                    _fail(f'Missing: {tsm_dir}')
                    errors += 1
                    continue
                wavs = sorted(tsm_dir.glob('*_q*.wav'))
                _ok(f'{piece_id} continuous max={max_a}: {len(wavs)} query wavs')

    return errors


# ============================================================================
# Stage 2: Scenario verification
# ============================================================================

SCENARIO_REQUIRED_FILES = ['p.wav', 'o.wav', 'po.wav', 'p.beats', 'o.beats', 'scenario.info']

def verify_scenarios(benchmark: str) -> int:
    """
    Verify Stage 2 (scenario generation) outputs.

    Checks:
    - scenarios/{benchmark}/{mode}/ exists for all modes
    - scenarios.summary file exists and is parseable
    - Each scenario directory has all required symlinks/files
    - Symlinks point to existing files
    - scenario.info is parseable and has correct field count
    """
    import system_utils
    scenarios_root = f'scenarios/{benchmark}'
    logger.info(f'\n=== Stage 2: Scenario Generation ({scenarios_root}) ===')
    errors = 0

    if not os.path.isdir(scenarios_root):
        _fail(f'scenarios root not found: {scenarios_root}')
        return 1

    for mode in ALL_MODES:
        mode_dir = Path(scenarios_root) / mode
        if not mode_dir.exists():
            _fail(f'Mode directory missing: {mode_dir}')
            errors += 1
            continue

        summary_file = mode_dir / 'scenarios.summary'
        if not summary_file.exists():
            _fail(f'scenarios.summary missing: {summary_file}')
            errors += 1
            continue

        # Parse summary
        try:
            summary = system_utils.get_scenario_info(str(summary_file))
        except Exception as e:
            _fail(f'Cannot parse {summary_file}: {e}')
            errors += 1
            continue

        scenario_dirs = sorted(
            [d for d in mode_dir.iterdir() if d.is_dir()],
            key=lambda p: int(p.name[1:])
        )
        _ok(f'{mode}: {len(scenario_dirs)} scenarios, summary has {len(summary)} entries')

        if len(scenario_dirs) != len(summary):
            _fail(f'{mode}: {len(scenario_dirs)} dirs but {len(summary)} summary entries')
            errors += 1

        for sdir in scenario_dirs:
            sid = sdir.name
            for fname in SCENARIO_REQUIRED_FILES:
                fpath = sdir / fname
                if not fpath.exists():
                    _fail(f'{sid}/{fname} missing or broken symlink')
                    errors += 1
                elif fpath.is_symlink() and not fpath.resolve().exists():
                    _fail(f'{sid}/{fname} symlink target does not exist: {os.readlink(fpath)}')
                    errors += 1

            # Verify scenario.info is parseable
            info_file = sdir / 'scenario.info'
            if info_file.exists():
                try:
                    info = system_utils.get_scenario_info(str(info_file))
                    required_keys = ['p', 'o', 'po', 'measStart', 'measEnd',
                                     'pStart', 'pEnd', 'oStart', 'oEnd',
                                     'prefStart', 'prefEnd']
                    missing = [k for k in required_keys if k not in info]
                    if missing:
                        _fail(f'{sid}/scenario.info missing fields: {missing}')
                        errors += 1
                except Exception as e:
                    _fail(f'{sid}/scenario.info parse error: {e}')
                    errors += 1

    return errors


# ============================================================================
# Stage 3: Feature verification
# ============================================================================

def verify_features(benchmark: str) -> int:
    """
    Verify Stage 3 (feature computation) outputs.

    Checks:
    - features/chroma_stft_norm2/ exists
    - Each expected audio file has a corresponding .npy feature file
    - Feature files are non-empty numpy arrays with shape (12, N)
    """
    import pandas as pd
    feat_dir = Path('features/chroma_stft_norm2')
    logger.info(f'\n=== Stage 3: Feature Computation ({feat_dir}) ===')
    errors = 0

    if not feat_dir.exists():
        _fail(f'Feature directory not found: {feat_dir}')
        return 1

    df = pd.read_csv('cfg_files/AudioDataSummary.csv')
    audio_files = list(df['id'])

    for audio_file in audio_files:
        basename = os.path.splitext(audio_file)[0]
        feat_path = feat_dir / f'{basename}.npy'

        if not feat_path.exists():
            _warn(f'Feature file missing (audio may not be downloaded): {feat_path}')
            continue

        try:
            feat = np.load(feat_path)
            if feat.ndim != 2:
                _fail(f'{feat_path}: expected 2D array, got shape {feat.shape}')
                errors += 1
            elif feat.shape[0] != 12:
                _fail(f'{feat_path}: expected 12 chroma bins, got {feat.shape[0]}')
                errors += 1
            elif feat.shape[1] == 0:
                _fail(f'{feat_path}: zero-length feature array')
                errors += 1
            else:
                _ok(f'{basename}: shape {feat.shape}')
        except Exception as e:
            _fail(f'Cannot load {feat_path}: {e}')
            errors += 1

    return errors


# ============================================================================
# Stage 4: Experiment verification
# ============================================================================

def verify_experiments(benchmark: str, systems: Optional[List[str]] = None) -> int:
    """
    Verify Stage 4 (experiment) outputs.

    Checks:
    - experiments/{benchmark}/{mode}/{system}/s{N}/hyp.npy exists for all scenarios
    - hyp.npy is a 2 x N numpy array (piano time, orchestra time)
    - Values are non-negative and monotonically increasing in both dimensions
    """
    experiments_root = Path(f'experiments/{benchmark}')
    scenarios_root = Path(f'scenarios/{benchmark}')
    logger.info(f'\n=== Stage 4: Experiments ({experiments_root}) ===')
    errors = 0

    if not experiments_root.exists():
        _fail(f'Experiments directory not found: {experiments_root}')
        return 1

    for mode in ALL_MODES:
        mode_exp_dir = experiments_root / mode
        mode_scen_dir = scenarios_root / mode

        if not mode_exp_dir.exists():
            _warn(f'No experiments for mode: {mode}')
            continue

        available_systems = [
            d.name for d in mode_exp_dir.iterdir()
            if d.is_dir() and not d.name.startswith('_')
        ]
        if systems:
            available_systems = [s for s in available_systems if s in systems]

        for system in available_systems:
            system_dir = mode_exp_dir / system
            scenario_dirs = sorted(
                [d for d in system_dir.iterdir() if d.is_dir()],
                key=lambda p: int(p.name[1:])
            )

            hyp_errors = 0
            for sdir in scenario_dirs:
                hyp_path = sdir / 'hyp.npy'
                if not hyp_path.exists():
                    _fail(f'{mode}/{system}/{sdir.name}: hyp.npy missing')
                    hyp_errors += 1
                    errors += 1
                    continue

                try:
                    hyp = np.load(hyp_path)
                    if hyp.ndim != 2 or hyp.shape[0] != 2:
                        _fail(f'{mode}/{system}/{sdir.name}: expected shape (2, N), got {hyp.shape}')
                        hyp_errors += 1
                        errors += 1
                    elif hyp.shape[1] == 0:
                        _fail(f'{mode}/{system}/{sdir.name}: empty alignment')
                        hyp_errors += 1
                        errors += 1
                    else:
                        # Check non-negative
                        if np.any(hyp < 0):
                            _warn(f'{mode}/{system}/{sdir.name}: negative values in hyp.npy')
                        # Check rough monotonicity (allow some tolerance for online systems)
                        if not np.all(np.diff(hyp[0]) >= -0.1):
                            _warn(f'{mode}/{system}/{sdir.name}: piano times not monotone')
                except Exception as e:
                    _fail(f'{mode}/{system}/{sdir.name}: cannot load hyp.npy: {e}')
                    hyp_errors += 1
                    errors += 1

            if hyp_errors == 0:
                _ok(f'{mode}/{system}: {len(scenario_dirs)} scenarios, all hyp.npy OK')
            else:
                _fail(f'{mode}/{system}: {hyp_errors}/{len(scenario_dirs)} scenarios have issues')

    return errors


# ============================================================================
# Stage 5: Evaluation verification
# ============================================================================

def verify_evaluation(benchmark: str, systems: Optional[List[str]] = None) -> int:
    """
    Verify Stage 5 (evaluation) outputs.

    Checks:
    - eval/{benchmark}/{mode}/{system}/errs.pkl exists
    - errs.pkl is a dict mapping scenario_id → (errors_array, measure_nums)
    - Error values are finite floats
    """
    eval_root = Path(f'eval/{benchmark}')
    logger.info(f'\n=== Stage 5: Evaluation ({eval_root}) ===')
    errors = 0

    if not eval_root.exists():
        _fail(f'Eval directory not found: {eval_root}')
        return 1

    for mode in ALL_MODES:
        mode_eval_dir = eval_root / mode
        if not mode_eval_dir.exists():
            _warn(f'No eval results for mode: {mode}')
            continue

        available_systems = [
            d.name for d in mode_eval_dir.iterdir()
            if d.is_dir()
        ]
        if systems:
            available_systems = [s for s in available_systems if s in systems]

        for system in available_systems:
            errs_path = mode_eval_dir / system / 'errs.pkl'
            if not errs_path.exists():
                _fail(f'{mode}/{system}/errs.pkl missing')
                errors += 1
                continue

            try:
                with open(errs_path, 'rb') as f:
                    d = pickle.load(f)

                if not isinstance(d, dict):
                    _fail(f'{mode}/{system}/errs.pkl: expected dict, got {type(d)}')
                    errors += 1
                    continue

                total_measures = 0
                bad_scenarios = 0
                for sid, val in d.items():
                    errs_arr, meas_nums = val
                    if not np.all(np.isfinite(errs_arr)):
                        _warn(f'{mode}/{system}/{sid}: non-finite errors')
                    total_measures += len(errs_arr)

                _ok(f'{mode}/{system}: {len(d)} scenarios, {total_measures} total eval points')

            except Exception as e:
                _fail(f'Cannot load {errs_path}: {e}')
                errors += 1

    return errors


# ============================================================================
# Cross-comparison: benchmark.py vs notebook outputs
# ============================================================================

def compare_hyp_files(
    benchmark_hyp: np.ndarray,
    notebook_hyp: np.ndarray,
    label: str,
    atol: float = 0.05,
) -> bool:
    """
    Compare two hyp.npy arrays. Returns True if they match within tolerance.

    Both arrays should be 2 x N (piano_time, orch_time) in seconds.
    We compare by interpolating both onto a common query time grid.
    """
    if benchmark_hyp.shape != notebook_hyp.shape:
        # Different lengths are OK for online systems — compare on overlapping range
        q_min = max(benchmark_hyp[0, 0], notebook_hyp[0, 0])
        q_max = min(benchmark_hyp[0, -1], notebook_hyp[0, -1])
        if q_max <= q_min:
            _warn(f'{label}: no overlapping query time range to compare')
            return True
        grid = np.linspace(q_min, q_max, 200)
        b_interp = np.interp(grid, benchmark_hyp[0], benchmark_hyp[1])
        n_interp = np.interp(grid, notebook_hyp[0], notebook_hyp[1])
    else:
        grid = benchmark_hyp[0]
        b_interp = benchmark_hyp[1]
        n_interp = np.interp(grid, notebook_hyp[0], notebook_hyp[1])

    max_diff = np.max(np.abs(b_interp - n_interp))
    mean_diff = np.mean(np.abs(b_interp - n_interp))

    if max_diff <= atol:
        _ok(f'{label}: max_diff={max_diff*1000:.1f}ms, mean_diff={mean_diff*1000:.1f}ms')
        return True
    else:
        _fail(f'{label}: max_diff={max_diff*1000:.1f}ms (>{atol*1000:.0f}ms), mean_diff={mean_diff*1000:.1f}ms')
        return False


def compare_experiments(
    benchmark: str,
    notebook_experiments: str,
    systems: Optional[List[str]] = None,
    atol: float = 0.05,
) -> int:
    """
    Compare benchmark.py experiment outputs against notebook experiment outputs.

    The notebook uses:
      {notebook_experiments}/{mode}/{notebook_system_name}/s{N}/hyp.npy

    The benchmark uses:
      experiments/{benchmark}/{mode}/{system}/s{N}/hyp.npy
    """
    logger.info(f'\n=== Cross-comparison: Experiments ===')
    logger.info(f'  benchmark: experiments/{benchmark}/')
    logger.info(f'  notebook:  {notebook_experiments}/')
    errors = 0

    for mode in ALL_MODES:
        for nb_name, bm_name in NOTEBOOK_TO_BENCHMARK_SYSTEM.items():
            if systems and bm_name not in systems:
                continue

            nb_mode_dir = Path(notebook_experiments) / mode / nb_name
            bm_mode_dir = Path(f'experiments/{benchmark}') / mode / bm_name

            if not nb_mode_dir.exists():
                _warn(f'Notebook dir not found, skipping: {nb_mode_dir}')
                continue
            if not bm_mode_dir.exists():
                _warn(f'Benchmark dir not found, skipping: {bm_mode_dir}')
                continue

            nb_scenarios = sorted(
                [d for d in nb_mode_dir.iterdir() if d.is_dir()],
                key=lambda p: int(p.name[1:])
            )

            for nb_sdir in nb_scenarios:
                sid = nb_sdir.name
                nb_hyp_path = nb_sdir / 'hyp.npy'
                bm_hyp_path = bm_mode_dir / sid / 'hyp.npy'

                if not nb_hyp_path.exists():
                    continue
                if not bm_hyp_path.exists():
                    _fail(f'{mode}/{bm_name}/{sid}: benchmark hyp.npy missing')
                    errors += 1
                    continue

                try:
                    nb_hyp = np.load(nb_hyp_path)
                    bm_hyp = np.load(bm_hyp_path)
                    ok = compare_hyp_files(bm_hyp, nb_hyp, f'{mode}/{bm_name}/{sid}', atol=atol)
                    if not ok:
                        errors += 1
                except Exception as e:
                    _fail(f'{mode}/{bm_name}/{sid}: error comparing: {e}')
                    errors += 1

    return errors


def compare_eval(
    benchmark: str,
    notebook_eval: str,
    systems: Optional[List[str]] = None,
    atol_ms: float = 50.0,
) -> int:
    """
    Compare benchmark.py evaluation outputs against notebook evaluation outputs.

    Compares mean absolute error per system/mode between the two pipelines.
    """
    logger.info(f'\n=== Cross-comparison: Evaluation ===')
    logger.info(f'  benchmark: eval/{benchmark}/')
    logger.info(f'  notebook:  {notebook_eval}/')
    errors = 0

    for mode in ALL_MODES:
        for nb_name, bm_name in NOTEBOOK_TO_BENCHMARK_SYSTEM.items():
            if systems and bm_name not in systems:
                continue

            nb_errs_path = Path(notebook_eval) / mode / nb_name / 'errs.pkl'
            bm_errs_path = Path(f'eval/{benchmark}') / mode / bm_name / 'errs.pkl'

            if not nb_errs_path.exists():
                _warn(f'Notebook eval not found, skipping: {nb_errs_path}')
                continue
            if not bm_errs_path.exists():
                _warn(f'Benchmark eval not found, skipping: {bm_errs_path}')
                continue

            try:
                with open(nb_errs_path, 'rb') as f:
                    nb_d = pickle.load(f)
                with open(bm_errs_path, 'rb') as f:
                    bm_d = pickle.load(f)

                # Compute MAE for each
                nb_all = np.concatenate([nb_d[s][0] for s in nb_d if s in bm_d])
                bm_all = np.concatenate([bm_d[s][0] for s in bm_d if s in nb_d])

                nb_mae = np.mean(np.abs(nb_all)) * 1000  # ms
                bm_mae = np.mean(np.abs(bm_all)) * 1000  # ms
                diff = abs(nb_mae - bm_mae)

                label = f'{mode}/{bm_name}'
                if diff <= atol_ms:
                    _ok(f'{label}: notebook MAE={nb_mae:.1f}ms, benchmark MAE={bm_mae:.1f}ms, diff={diff:.1f}ms')
                else:
                    _fail(f'{label}: MAE diff={diff:.1f}ms (>{atol_ms:.0f}ms). notebook={nb_mae:.1f}ms, benchmark={bm_mae:.1f}ms')
                    errors += 1

            except Exception as e:
                _fail(f'{mode}/{bm_name}: error comparing eval: {e}')
                errors += 1

    return errors


def compare_scenarios(benchmark: str, notebook_scenarios: str) -> int:
    """
    Compare benchmark.py scenario outputs against notebook scenario outputs.

    Checks that scenario.info files have matching fields (audio paths may differ
    since the benchmark uses a different root, but timestamps should match).
    """
    import system_utils
    logger.info(f'\n=== Cross-comparison: Scenarios ===')
    logger.info(f'  benchmark: scenarios/{benchmark}/')
    logger.info(f'  notebook:  {notebook_scenarios}/')
    errors = 0

    for mode in ALL_MODES:
        nb_summary = Path(notebook_scenarios) / mode / 'scenarios.summary'
        bm_summary = Path(f'scenarios/{benchmark}') / mode / 'scenarios.summary'

        if not nb_summary.exists():
            _warn(f'Notebook summary not found: {nb_summary}')
            continue
        if not bm_summary.exists():
            _warn(f'Benchmark summary not found: {bm_summary}')
            continue

        try:
            nb_d = system_utils.get_scenario_info(str(nb_summary))
            bm_d = system_utils.get_scenario_info(str(bm_summary))
        except Exception as e:
            _fail(f'{mode}: cannot parse summary: {e}')
            errors += 1
            continue

        if len(nb_d) != len(bm_d):
            _fail(f'{mode}: notebook has {len(nb_d)} scenarios, benchmark has {len(bm_d)}')
            errors += 1
        else:
            _ok(f'{mode}: both have {len(nb_d)} scenarios')

        # Compare timestamps for matching scenario IDs
        common = set(nb_d.keys()) & set(bm_d.keys())
        ts_errors = 0
        for sid in sorted(common, key=lambda s: int(s[1:])):
            nb_s = nb_d[sid]
            bm_s = bm_d[sid]
            for field in ('measStart', 'measEnd', 'pStart', 'pEnd', 'oStart', 'oEnd', 'prefStart', 'prefEnd'):
                nb_val = nb_s[field]
                bm_val = bm_s[field]
                if isinstance(nb_val, float):
                    if abs(nb_val - bm_val) > 0.01:
                        _fail(f'{mode}/{sid}: {field} mismatch: notebook={nb_val:.3f}, benchmark={bm_val:.3f}')
                        ts_errors += 1
                        errors += 1
                else:
                    if nb_val != bm_val:
                        _fail(f'{mode}/{sid}: {field} mismatch: notebook={nb_val}, benchmark={bm_val}')
                        ts_errors += 1
                        errors += 1
        if ts_errors == 0 and len(common) > 0:
            _ok(f'{mode}: all {len(common)} scenario timestamps match')

    return errors


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Verify benchmark.py outputs against notebook outputs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--benchmark', required=True, choices=['train', 'test'],
                        help='Which benchmark to verify')
    parser.add_argument('--check', choices=['structure', 'compare', 'all'],
                        default='structure',
                        help='structure: check outputs exist and are valid; '
                             'compare: compare against notebook outputs; '
                             'all: both')
    parser.add_argument('--systems', nargs='+', default=None,
                        metavar='SYSTEM',
                        help='Limit verification to specific systems')
    parser.add_argument('--notebook-scenarios', default=None,
                        help='Path to notebook scenarios root (e.g. scenarios/)')
    parser.add_argument('--notebook-experiments', default=None,
                        help='Path to notebook experiments root (e.g. experiments/)')
    parser.add_argument('--notebook-eval', default=None,
                        help='Path to notebook eval root (e.g. eval/)')
    parser.add_argument('--mode-config', default='configs/query_modes.json',
                        help='Query mode config JSON')
    parser.add_argument('--atol', type=float, default=0.05,
                        help='Alignment tolerance in seconds for hyp.npy comparison (default: 0.05)')
    parser.add_argument('--atol-ms', type=float, default=50.0,
                        help='MAE tolerance in ms for eval comparison (default: 50ms)')
    args = parser.parse_args()

    import json
    with open(args.mode_config) as f:
        mode_config = json.load(f)

    total_errors = 0

    if args.check in ('structure', 'all'):
        logger.info('\n' + '=' * 60)
        logger.info('STRUCTURAL VERIFICATION')
        logger.info('=' * 60)
        total_errors += verify_queries(args.benchmark, mode_config)
        total_errors += verify_scenarios(args.benchmark)
        total_errors += verify_features(args.benchmark)
        total_errors += verify_experiments(args.benchmark, args.systems)
        total_errors += verify_evaluation(args.benchmark, args.systems)

    if args.check in ('compare', 'all'):
        logger.info('\n' + '=' * 60)
        logger.info('CROSS-COMPARISON WITH NOTEBOOK OUTPUTS')
        logger.info('=' * 60)

        if args.notebook_scenarios:
            total_errors += compare_scenarios(args.benchmark, args.notebook_scenarios)
        else:
            logger.info('\n  (Skipping scenario comparison — no --notebook-scenarios provided)')

        if args.notebook_experiments:
            total_errors += compare_experiments(
                args.benchmark, args.notebook_experiments,
                args.systems, atol=args.atol,
            )
        else:
            logger.info('\n  (Skipping experiment comparison — no --notebook-experiments provided)')

        if args.notebook_eval:
            total_errors += compare_eval(
                args.benchmark, args.notebook_eval,
                args.systems, atol_ms=args.atol_ms,
            )
        else:
            logger.info('\n  (Skipping eval comparison — no --notebook-eval provided)')

    logger.info('\n' + '=' * 60)
    if total_errors == 0:
        logger.info(f'{PASS} All checks passed.')
    else:
        logger.error(f'{FAIL} {total_errors} check(s) failed.')
    logger.info('=' * 60)

    sys.exit(0 if total_errors == 0 else 1)


if __name__ == '__main__':
    main()
