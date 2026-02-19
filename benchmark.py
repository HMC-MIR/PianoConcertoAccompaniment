#!/usr/bin/env python3
"""
PianoConcertoAccompaniment Benchmark Pipeline

Command-line interface for running the benchmark pipeline in stages or all at once.
All stages can be run independently or together with the 'run' command.

Stages:
  1. queries    - Generate time-scaled audio queries
  2. scenarios  - Generate alignment scenario directories
  3. features   - Precompute chroma STFT features
  4. experiment - Run alignment systems on all scenarios
  5. evaluate   - Compute alignment errors

Usage examples:
  # Run full pipeline (train, all modes, all systems)
  python benchmark.py run --benchmark train --systems DTW NOA NOA_MONOTONIC OLTW OLTW_GLOBAL

  # Run individual stages
  python benchmark.py queries   --benchmark train --mode constant
  python benchmark.py scenarios --benchmark train --mode constant
  python benchmark.py features  --benchmark train
  python benchmark.py experiment --benchmark train --systems DTW NOA
  python benchmark.py evaluate   --benchmark train

  # Use custom config files
  python benchmark.py experiment --benchmark train --systems OLTW_GLOBAL \\
      --system-config configs/oltw_global_examples.json
"""

import os
import re
import sys
import json
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import librosa as lb
from tqdm import tqdm

# Local imports
import system_utils
import eval_tools
from utils.query import QueryGenerator
from utils.query.scenario import (
    generateScenariosConstant,
    generateScenariosRandom,
    generateScenariosContinuous,
    myLogger,
)
from utils.query.utils import get_audio_files


# ============================================================================
# Paths and Constants
# ============================================================================

TRAIN_LIST_FILE = 'cfg_files/train.list'
TEST_LIST_FILE = 'cfg_files/test.list'
AUDIO_SUMMARY_FILE = 'cfg_files/AudioDataSummary.csv'
AUDIO_ROOT = 'audio'
ANNOT_ROOT = 'annot'
QUERY_MEASURES_FILE = 'annot/query.measures'

DEFAULT_SYSTEM_CONFIG = 'configs/default_systems.json'
DEFAULT_MODE_CONFIG = 'configs/query_modes.json'

ALL_MODES = ['constant', 'random', 'continuous']
ALL_SYSTEMS = ['DTW', 'NOA', 'NOA_MONOTONIC', 'OLTW', 'OLTW_GLOBAL']

# Seeds per benchmark type
BENCHMARK_SEEDS = {
    'train': {'num_queries': 1, 'seeds': list(range(1))}, # TODO: change from 1 to 5
    'test':  {'num_queries': 10, 'seeds': list(range(100, 110))},
}


def get_queries_root(benchmark: str) -> str:
    return f'queries/{benchmark}'


def get_scenarios_root(benchmark: str) -> str:
    return f'scenarios/{benchmark}'


def get_experiments_root(benchmark: str) -> str:
    return f'experiments/{benchmark}'


def get_eval_root(benchmark: str) -> str:
    return f'eval/{benchmark}'


def get_features_root() -> str:
    return 'features/chroma_stft_norm2'


# ============================================================================
# Logging
# ============================================================================

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """Set up logging to both a timestamped file and the console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'benchmark_{timestamp}.log')

    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'Logging to {log_file}')
    return logger


# ============================================================================
# Configuration helpers
# ============================================================================

def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def get_piece_ids(list_file: str) -> List[str]:
    """Parse a train.list or test.list file and return piece IDs."""
    ids = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def get_fullmix_file(piece_id: str) -> str:
    """Return the PO1 wav filename for a given piece_id."""
    for fname in get_audio_files(AUDIO_SUMMARY_FILE, f'^{piece_id}_PO1\\.\\S+$'):
        return re.sub(r'\.mp3$', '.wav', fname)
    raise FileNotFoundError(f'No PO1 full-mix file found for {piece_id}')


def prepare_system_kwargs(system: str, system_config: dict) -> dict:
    """Convert JSON config values to numpy arrays where needed."""
    kwargs = system_config.copy()
    for key in ('steps', 'DTW_steps', 'window_steps'):
        if key in kwargs:
            kwargs[key] = np.array(kwargs[key])
    if 'weights' in kwargs:
        kwargs['weights'] = np.array(kwargs['weights'])
    if 'DTW_weights' in kwargs:
        kwargs['DTW_weights'] = np.array(kwargs['DTW_weights'])
    return kwargs


# ============================================================================
# Stage 1: Query Generation
# ============================================================================

def cmd_queries(args, logger: logging.Logger):
    """Stage 1: Generate time-scaled audio queries."""
    benchmark = args.benchmark
    modes = [args.mode] if args.mode else ALL_MODES
    mode_config = load_json(args.mode_config)

    queries_root = get_queries_root(benchmark)
    list_file = TRAIN_LIST_FILE if benchmark == 'train' else TEST_LIST_FILE

    if not os.path.exists(list_file):
        logger.error(f'List file not found: {list_file}')
        sys.exit(1)

    num_queries = BENCHMARK_SEEDS[benchmark]['num_queries']

    query_generator = QueryGenerator(
        audio_summary_file=AUDIO_SUMMARY_FILE,
        query_measures_file=QUERY_MEASURES_FILE,
        audio_root=AUDIO_ROOT,
        annot_root=ANNOT_ROOT,
    )

    logger.info(f'Generating queries for benchmark={benchmark}, modes={modes}')

    for mode in modes:
        logger.info(f'  Mode: {mode}')
        cfg = mode_config[mode]

        try:
            if mode == 'constant':
                tsm_factors = cfg['tsm_factors']
                logger.info(f'    tsm_factors={tsm_factors}')
                query_generator.generateQueriesConstant(queries_root, tsm_factors)

            elif mode == 'random':
                max_tsm_factor = cfg['max_tsm_factor']
                logger.info(f'    max_tsm_factor={max_tsm_factor}, num_queries={num_queries}')
                query_generator.generateQueriesRandom(queries_root, max_tsm_factor, num_queries)

            elif mode == 'continuous':
                max_alpha_change = cfg['max_alpha_change']
                logger.info(f'    max_alpha_change={max_alpha_change}, num_queries={num_queries}')
                query_generator.generateQueriesContinuous(queries_root, max_alpha_change, num_queries)

        except Exception as e:
            logger.error(f'Error generating {mode} queries: {e}', exc_info=True)
            raise

        logger.info(f'  {mode} queries complete.')

    logger.info('Query generation complete.')
    _verify_queries(queries_root, modes, mode_config, num_queries, logger)


def _verify_queries(queries_root, modes, mode_config, num_queries, logger):
    """Verify generated query files are non-empty and counts are correct."""
    logger.info('Verifying generated queries...')
    errors = 0

    for piece_dir in Path(queries_root).iterdir():
        if not piece_dir.is_dir():
            continue
        for mode in modes:
            cfg = mode_config[mode]
            if mode == 'constant':
                for tsm in cfg['tsm_factors']:
                    tsm_dir = piece_dir / f'tsm{tsm}'
                    for f in tsm_dir.glob('*.wav'):
                        if f.stat().st_size == 0:
                            logger.warning(f'Empty wav file: {f}')
                            errors += 1
            elif mode == 'random':
                max_f = cfg['max_tsm_factor']
                tsm_dir = piece_dir / f'tsm_random_max{max_f}'
                wavs = list(tsm_dir.glob('*_q*.wav')) if tsm_dir.exists() else []
                if len(wavs) == 0:
                    logger.warning(f'No query wavs found in {tsm_dir}')
                    errors += 1
            elif mode == 'continuous':
                max_a = cfg['max_alpha_change']
                tsm_dir = piece_dir / f'tsm_continuous_max{max_a}'
                wavs = list(tsm_dir.glob('*_q*.wav')) if tsm_dir.exists() else []
                if len(wavs) == 0:
                    logger.warning(f'No query wavs found in {tsm_dir}')
                    errors += 1

    if errors == 0:
        logger.info('Query verification passed.')
    else:
        logger.warning(f'Query verification found {errors} issue(s).')


# ============================================================================
# Stage 2: Scenario Generation
# ============================================================================

def cmd_scenarios(args, logger: logging.Logger):
    """Stage 2: Generate alignment scenario directories."""
    benchmark = args.benchmark
    modes = [args.mode] if args.mode else ALL_MODES
    mode_config = load_json(args.mode_config)

    queries_root = get_queries_root(benchmark)
    scenarios_root = get_scenarios_root(benchmark)
    list_file = TRAIN_LIST_FILE if benchmark == 'train' else TEST_LIST_FILE

    if not os.path.exists(list_file):
        logger.error(f'List file not found: {list_file}')
        sys.exit(1)

    num_queries = BENCHMARK_SEEDS[benchmark]['num_queries']

    # Regenerate from scratch
    if os.path.exists(scenarios_root):
        logger.info(f'Removing existing scenarios directory: {scenarios_root}')
        shutil.rmtree(scenarios_root)
    os.makedirs(scenarios_root)

    for mode in ALL_MODES:
        os.makedirs(os.path.join(scenarios_root, mode))

    cnts = {m: 0 for m in ALL_MODES}
    log_info = {m: [] for m in ALL_MODES}

    piece_ids = get_piece_ids(list_file)
    logger.info(f'Generating scenarios for {len(piece_ids)} pieces, modes={modes}')

    for piece_id in piece_ids:
        logger.info(f'  Processing piece: {piece_id}')
        try:
            fullmix_file = get_fullmix_file(piece_id)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue

        common_kwargs = dict(
            piece_id=piece_id,
            fullmix_file=fullmix_file,
            QUERIES_ROOT=queries_root,
            ANNOT_ROOT=ANNOT_ROOT,
            AUDIO_ROOT=AUDIO_ROOT,
            QUERY_MEASURES_FILE=QUERY_MEASURES_FILE,
        )

        for mode in modes:
            cfg = mode_config[mode]
            outdir = os.path.join(scenarios_root, mode)

            try:
                if mode == 'constant':
                    res = generateScenariosConstant(
                        cnt=cnts[mode],
                        tsm_factors=cfg['tsm_factors'],
                        outdir=outdir,
                        **common_kwargs,
                    )
                elif mode == 'random':
                    res = generateScenariosRandom(
                        cnt=cnts[mode],
                        num_queries=num_queries,
                        max_tsm_factors=[cfg['max_tsm_factor']],
                        outdir=outdir,
                        **common_kwargs,
                    )
                elif mode == 'continuous':
                    res = generateScenariosContinuous(
                        cnt=cnts[mode],
                        num_queries=num_queries,
                        max_alpha_changes=[cfg['max_alpha_change']],
                        outdir=outdir,
                        **common_kwargs,
                    )

                cnts[mode] = res['cnt']
                log_info[mode].extend(res['logInfo'])

            except Exception as e:
                logger.error(
                    f'Error generating {mode} scenarios for {piece_id}: {e}',
                    exc_info=True,
                )

    # Write summary files
    for mode in ALL_MODES:
        summary_file = os.path.join(scenarios_root, mode, 'scenarios.summary')
        if log_info[mode]:
            myLogger(summary_file, log_info[mode])
            logger.info(f'  {mode}: {cnts[mode]} scenarios → {summary_file}')
        else:
            logger.info(f'  {mode}: 0 scenarios generated (mode may not have been requested)')

    logger.info('Scenario generation complete.')


# ============================================================================
# Stage 3: Feature Computation
# ============================================================================

def cmd_features(args, logger: logging.Logger):
    """Stage 3: Precompute chroma STFT features for all audio files."""
    benchmark = args.benchmark
    list_file = TRAIN_LIST_FILE if benchmark == 'train' else TEST_LIST_FILE

    if not os.path.exists(list_file):
        logger.error(f'List file not found: {list_file}')
        sys.exit(1)

    piece_ids = get_piece_ids(list_file)
    feat_dir = get_features_root()
    os.makedirs(feat_dir, exist_ok=True)

    # Collect all audio files to compute features for (P and O recordings)
    import pandas as pd
    df = pd.read_csv(AUDIO_SUMMARY_FILE)
    audio_files = list(df['id'])

    logger.info(f'Computing chroma STFT features for {len(audio_files)} audio files')

    hop_length = 512
    sr_target = 22050

    for audio_file in tqdm(audio_files, desc='Computing features'):
        basename = os.path.splitext(audio_file)[0]
        feat_path = os.path.join(feat_dir, f'{basename}.npy')

        if os.path.exists(feat_path):
            logger.debug(f'Skipping {basename} — already computed')
            continue

        audio_path = os.path.join(AUDIO_ROOT, f'{basename}.wav')
        if not os.path.exists(audio_path):
            logger.warning(f'Audio file not found, skipping: {audio_path}')
            continue

        try:
            y, sr = lb.load(audio_path, sr=sr_target)
            feat = lb.feature.chroma_stft(
                y=y, sr=sr,
                hop_length=hop_length,
                center=False,
                norm=2,
            )
            np.save(feat_path, feat)
            logger.debug(f'Saved features: {feat_path}')
        except Exception as e:
            logger.error(f'Error computing features for {audio_file}: {e}', exc_info=True)

    logger.info('Feature computation complete.')


# ============================================================================
# Stage 4: Run Experiments
# ============================================================================

def cmd_experiment(args, logger: logging.Logger):
    """Stage 4: Run alignment systems on all scenarios for all modes."""
    benchmark = args.benchmark
    systems = args.systems
    system_configs = load_json(args.system_config)
    scenarios_root = get_scenarios_root(benchmark)
    experiments_root = get_experiments_root(benchmark)

    if not os.path.exists(scenarios_root):
        logger.error(f'Scenarios directory not found: {scenarios_root}')
        logger.error("Please run 'scenarios' stage first.")
        sys.exit(1)

    logger.info(f'Running experiments: benchmark={benchmark}, systems={systems}')

    for mode in ALL_MODES:
        mode_scenarios_dir = os.path.join(scenarios_root, mode)
        if not os.path.isdir(mode_scenarios_dir):
            logger.warning(f'Mode directory not found, skipping: {mode_scenarios_dir}')
            continue

        scenario_dirs = sorted(
            [d for d in Path(mode_scenarios_dir).iterdir() if d.is_dir()],
            key=lambda p: int(p.name[1:])  # sort s1, s2, ... numerically
        )
        if not scenario_dirs:
            logger.warning(f'No scenarios found in {mode_scenarios_dir}')
            continue

        logger.info(f'  Mode: {mode} ({len(scenario_dirs)} scenarios)')

        for system in systems:
            if system not in system_configs:
                logger.error(f'No configuration found for system: {system}. Skipping.')
                continue

            kwargs = prepare_system_kwargs(system, system_configs[system])
            logger.info(f'    System: {system}')

            for scenario_dir in tqdm(scenario_dirs, desc=f'{mode}/{system}'):
                scenario_id = scenario_dir.name
                out_dir = os.path.join(experiments_root, mode, system, scenario_id)

                if os.path.exists(out_dir):
                    logger.debug(f'Skipping {out_dir} — already exists')
                    continue

                # Build per-scenario cache dir (shared across systems that use same features)
                cache_dir = os.path.join(experiments_root, mode, '_cache', scenario_id)

                try:
                    _run_offline(system, str(scenario_dir), cache_dir, kwargs, logger)
                    _run_online(system, str(scenario_dir), out_dir, cache_dir, kwargs, logger)
                except Exception as e:
                    logger.error(
                        f'Error running {system} on {scenario_id} ({mode}): {e}',
                        exc_info=True,
                    )
                    continue

    logger.info('Experiment stage complete.')


def _run_offline(system, scenario_dir, cache_dir, kwargs, logger):
    """Dispatch offline processing to the appropriate system module."""
    hop_length = kwargs.get('hop_length', 512)

    if system == 'DTW':
        from utils.systems import dtw
        dtw.offline_processing(scenario_dir, cache_dir, hop_length)

    elif system in ('NOA', 'NOA_MONOTONIC'):
        from utils.systems import noa
        noa.offline_processing(scenario_dir, cache_dir, hop_length)

    elif system == 'OLTW':
        from utils.systems import oltw
        oltw.offline_processing(scenario_dir, cache_dir, hop_length)

    elif system == 'OLTW_GLOBAL':
        from utils.systems import oltw_global
        oltw_global.offline_processing(scenario_dir, cache_dir, hop_length)

    else:
        raise ValueError(f'Unknown system: {system}')


def _run_online(system, scenario_dir, out_dir, cache_dir, kwargs, logger):
    """Dispatch online processing to the appropriate system module."""
    hop_length = kwargs.get('hop_length', 512)

    if system == 'DTW':
        from utils.systems import dtw
        dtw.online_processing(
            scenario_dir, out_dir, cache_dir,
            hop_length=hop_length,
            steps=kwargs['steps'],
            weights=kwargs['weights'],
        )

    elif system in ('NOA', 'NOA_MONOTONIC'):
        from utils.systems import noa
        noa.online_processing(
            scenario_dir, out_dir, cache_dir,
            hop_length=hop_length,
            monotonic=(system == 'NOA_MONOTONIC'),
        )

    elif system == 'OLTW':
        from utils.systems import oltw
        try:
            oltw.online_processing(
                scenario_dir, out_dir, cache_dir,
                hop_length=hop_length,
                jar_path=kwargs.get('jar_path'),
            )
        except (RuntimeError, FileNotFoundError) as e:
            logger.error(f'OLTW skipped ({scenario_dir}): {e}')
            raise

    elif system == 'OLTW_GLOBAL':
        from utils.systems import oltw_global
        oltw_global.online_processing(
            scenario_dir, out_dir, cache_dir,
            hop_length=hop_length,
            c=kwargs.get('c'),
            DTW_steps=kwargs['DTW_steps'],
            DTW_weights=kwargs['DTW_weights'],
            window_steps=kwargs['window_steps'],
        )

    else:
        raise ValueError(f'Unknown system: {system}')


# ============================================================================
# Stage 5: Evaluation
# ============================================================================

def cmd_evaluate(args, logger: logging.Logger):
    """Stage 5: Compute alignment errors for all modes and systems."""
    benchmark = args.benchmark
    scenarios_root = get_scenarios_root(benchmark)
    experiments_root = get_experiments_root(benchmark)
    eval_root = get_eval_root(benchmark)

    if not os.path.exists(experiments_root):
        logger.error(f'Experiments directory not found: {experiments_root}')
        logger.error("Please run 'experiment' stage first.")
        sys.exit(1)

    logger.info(f'Evaluating benchmark={benchmark}')

    for mode in ALL_MODES:
        mode_exp_dir = os.path.join(experiments_root, mode)
        mode_scenarios_dir = os.path.join(scenarios_root, mode)

        if not os.path.isdir(mode_exp_dir):
            logger.warning(f'No experiments found for mode {mode}, skipping.')
            continue

        if not os.path.isdir(mode_scenarios_dir):
            logger.warning(f'No scenarios found for mode {mode}, skipping.')
            continue

        # Find all system subdirectories (skip _cache)
        systems = [
            d for d in os.listdir(mode_exp_dir)
            if os.path.isdir(os.path.join(mode_exp_dir, d)) and not d.startswith('_')
        ]

        if not systems:
            logger.warning(f'No system results found in {mode_exp_dir}')
            continue

        logger.info(f'  Mode: {mode}, systems: {systems}')

        for system in systems:
            system_exp_dir = os.path.join(mode_exp_dir, system)
            system_eval_dir = os.path.join(eval_root, mode, system)

            logger.info(f'    Evaluating {system}...')
            try:
                eval_tools.calcAlignErrors_batch(
                    system_exp_dir,
                    mode_scenarios_dir,
                    system_eval_dir,
                )
                logger.info(f'    {system} evaluation complete → {system_eval_dir}')
            except Exception as e:
                logger.error(f'Error evaluating {system} ({mode}): {e}', exc_info=True)

    logger.info('Evaluation complete.')


# ============================================================================
# Full Pipeline
# ============================================================================

def cmd_run(args, logger: logging.Logger):
    """Run all stages in sequence."""
    sep = '=' * 70
    logger.info(sep)
    logger.info(f'FULL PIPELINE: benchmark={args.benchmark}, systems={args.systems}')
    logger.info(sep)

    logger.info('\n' + sep)
    logger.info('STAGE 1: Query Generation')
    logger.info(sep)
    cmd_queries(args, logger)

    logger.info('\n' + sep)
    logger.info('STAGE 2: Scenario Generation')
    logger.info(sep)
    cmd_scenarios(args, logger)

    logger.info('\n' + sep)
    logger.info('STAGE 3: Feature Computation')
    logger.info(sep)
    cmd_features(args, logger)

    logger.info('\n' + sep)
    logger.info('STAGE 4: Experiments')
    logger.info(sep)
    cmd_experiment(args, logger)

    logger.info('\n' + sep)
    logger.info('STAGE 5: Evaluation')
    logger.info(sep)
    cmd_evaluate(args, logger)

    logger.info('\n' + sep)
    logger.info('PIPELINE COMPLETE')
    logger.info(sep)


# ============================================================================
# CLI
# ============================================================================

def add_common_args(parser):
    parser.add_argument(
        '--benchmark', required=True, choices=['train', 'test'],
        help='Which benchmark to run (train or test)',
    )


def add_mode_arg(parser):
    parser.add_argument(
        '--mode', choices=ALL_MODES, default=None,
        help='Query generation mode. If omitted, runs all modes.',
    )


def add_config_args(parser):
    parser.add_argument(
        '--system-config', default=DEFAULT_SYSTEM_CONFIG,
        help=f'JSON config for system parameters (default: {DEFAULT_SYSTEM_CONFIG})',
    )
    parser.add_argument(
        '--mode-config', default=DEFAULT_MODE_CONFIG,
        help=f'JSON config for query mode parameters (default: {DEFAULT_MODE_CONFIG})',
    )


def add_systems_arg(parser):
    parser.add_argument(
        '--systems', nargs='+', required=True,
        metavar='SYSTEM',
        help=f'Alignment systems to run. Choices: {ALL_SYSTEMS}',
    )


def main():
    parser = argparse.ArgumentParser(
        description='PianoConcertoAccompaniment Benchmark Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest='command', help='Pipeline stage to run')
    subparsers.required = True

    # --- queries ---
    p = subparsers.add_parser('queries', help='Stage 1: Generate audio queries')
    add_common_args(p)
    add_mode_arg(p)
    add_config_args(p)

    # --- scenarios ---
    p = subparsers.add_parser('scenarios', help='Stage 2: Generate alignment scenarios')
    add_common_args(p)
    add_mode_arg(p)
    add_config_args(p)

    # --- features ---
    p = subparsers.add_parser('features', help='Stage 3: Precompute chroma STFT features')
    add_common_args(p)

    # --- experiment ---
    p = subparsers.add_parser('experiment', help='Stage 4: Run alignment systems')
    add_common_args(p)
    add_systems_arg(p)
    add_config_args(p)

    # --- evaluate ---
    p = subparsers.add_parser('evaluate', help='Stage 5: Compute alignment errors')
    add_common_args(p)

    # --- run (full pipeline) ---
    p = subparsers.add_parser('run', help='Run all stages in sequence')
    add_common_args(p)
    add_mode_arg(p)
    add_systems_arg(p)
    add_config_args(p)

    args = parser.parse_args()
    logger = setup_logging()

    # Attach defaults for args that may not exist on all subcommands
    if not hasattr(args, 'mode'):
        args.mode = None
    if not hasattr(args, 'systems'):
        args.systems = ALL_SYSTEMS
    if not hasattr(args, 'system_config'):
        args.system_config = DEFAULT_SYSTEM_CONFIG
    if not hasattr(args, 'mode_config'):
        args.mode_config = DEFAULT_MODE_CONFIG

    try:
        dispatch = {
            'queries': cmd_queries,
            'scenarios': cmd_scenarios,
            'features': cmd_features,
            'experiment': cmd_experiment,
            'evaluate': cmd_evaluate,
            'run': cmd_run,
        }
        dispatch[args.command](args, logger)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f'Fatal error: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
