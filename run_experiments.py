import os
import sys

# ensure workspace modules are importable
sys.path.append(os.getcwd())

import import_ipynb  # allow importing notebook files as modules
import numpy as np
import system_utils
from pathlib import Path
from multiprocessing import Pool
import logging
from datetime import datetime
from online_alignment.constants import OLTW_STEPS, OLTW_WEIGHTS

# configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ensure logs directory exists and add a file handler (timestamped)
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logfile = os.path.join(LOG_DIR, f'run_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler = logging.FileHandler(logfile)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(file_handler)


def getCacheDir(d, scenario_id, CACHE_ROOT_DIR):
    '''
    Returns the filepath of the cache directory for the given scenario id.  Note that the cache directory
    can be shared as long as the orchestra and full mix files match, so its naming specifies information
    from both.
    
    Inputs
    d: dictionary summarizing the information in the scenarios.summary file
    scenario_id: the identifier of the scenario of interest (e.g. s1)
    '''
    
    o_id = os.path.splitext(os.path.basename(d[scenario_id]['o']))[0] # e.g. rach2_mov1_O1
    p_id = '_'.join(o_id.split('_')[:-1]) + '_P1' # e.g. rach2_mov1_P1
    cache_id = p_id # e.g. rach2_mov1_P1
    cache_dir = f'{CACHE_ROOT_DIR}/{cache_id}' 
    
    return cache_dir

def runOfflineProcessing(system,
                          EXP_NAME,
                          SCENARIOS_ROOT_DIR,
                          EXP_ROOT_DIR,
                          CACHE_ROOT_DIR,
                          SCENARIOS_SUMMARY,
                          hop_size,
                          dtw_steps=None,
                          dtw_weights=None,
                          cache=None,
                          test_scenario=None):            # added parameter
    '''
    Runs offline processing for all scenarios of a given experiment configuration.
    '''
    logger.info(f'OfflineProcessing start: {EXP_NAME} | scenarios={SCENARIOS_ROOT_DIR} -> exp={EXP_ROOT_DIR}')
    # prepare directories
    if not os.path.exists(EXP_ROOT_DIR):
        os.makedirs(EXP_ROOT_DIR)
    if cache is None:
        if not os.path.exists(CACHE_ROOT_DIR):
            os.mkdir(CACHE_ROOT_DIR)
    else:
        if os.path.exists(CACHE_ROOT_DIR):
            os.system(f'rm -rf {CACHE_ROOT_DIR}')
        os.system(f'cp -r {cache} {CACHE_ROOT_DIR}')

    d = system_utils.get_scenario_info(SCENARIOS_SUMMARY)

    # restrict to one scenario if requested
    if test_scenario is not None:
        indices = [test_scenario - 1]
    else:
        indices = range(len(d))

    for i in indices:
        scenario_id = f's{i+1}'
        scenario_dir = f'{SCENARIOS_ROOT_DIR}/{scenario_id}'
        cache_dir = getCacheDir(d, scenario_id, CACHE_ROOT_DIR)
        logger.info(f'OfflineProcessing: {EXP_NAME} - processing {scenario_id} -> cache {cache_dir}')

        if EXP_NAME in ('DTW', 'NOA', 'NOA-Mono'):
            system.offline_processing(scenario_dir, cache_dir, hop_size)
        elif EXP_NAME in ('OLTW', 'OLTW-Global'):
            system.offline_processing(scenario_dir, cache_dir, hop_size)
        else:
            raise ValueError(f'Unknown experiment name: {EXP_NAME}')
        logger.info(f'OfflineProcessing: {EXP_NAME} - finished {scenario_id}')

def singleFileOnline(id,
                     d,
                     system,
                     EXP_NAME,
                     SCENARIOS_ROOT_DIR,
                     EXP_ROOT_DIR,
                     CACHE_ROOT_DIR,
                     hop_size,
                     dtw_steps=None,
                     dtw_weights=None,
                     jar_path=None,
                     copydir=None):
    scenario_id = f's{id+1}'
    scenario_dir = f'{SCENARIOS_ROOT_DIR}/{scenario_id}'
    out_dir = f'{EXP_ROOT_DIR}/{scenario_id}'
    cache_dir = getCacheDir(d, scenario_id, CACHE_ROOT_DIR)

    logger.info(f'SingleFileOnline: {EXP_NAME} {scenario_id} start')

    if os.path.exists(out_dir) and os.path.exists(f'{out_dir}/hyp.npy'):
        logger.info(f'SingleFileOnline: {scenario_id} already has hyp.npy, skipping')
        return

    if copydir is not None:
        src_dir = f'{copydir}/{scenario_id}'
        if os.path.exists(src_dir):
            logger.info(f'SingleFileOnline: copying from {src_dir} to {out_dir}')
            os.system(f'cp -r {src_dir} {out_dir}')
            return

    if EXP_NAME == 'DTW':
        logger.info(f'SingleFileOnline: running DTW online_processing for {scenario_id}')
        system.online_processing(scenario_dir, out_dir, cache_dir, hop_size, dtw_steps, dtw_weights)
    elif EXP_NAME == 'NOA':
        logger.info(f'SingleFileOnline: running NOA online_processing for {scenario_id}')
        system.online_processing(scenario_dir, out_dir, cache_dir, hop_size)
    elif EXP_NAME == 'OLTW':
        logger.info(f'SingleFileOnline: running OLTW online_processing for {scenario_id}')
        system.online_processing(scenario_dir, out_dir, cache_dir, hop_size, jar_path=jar_path)
    elif EXP_NAME == 'NOA-Mono':
        logger.info(f'SingleFileOnline: running NOA-Mono online_processing for {scenario_id}')
        system.online_processing(scenario_dir, out_dir, cache_dir, hop_size, monotonic=True)
    elif EXP_NAME == 'OLTW-Global':
        logger.info(f'SingleFileOnline: running OLTW-Global online_processing for {scenario_id}')
        system.online_processing(scenario_dir, out_dir, cache_dir, hop_size, c=None)
    else:
        raise ValueError(f'Unknown experiment name: {EXP_NAME}')
    logger.info(f'SingleFileOnline: {EXP_NAME} {scenario_id} finished')
    
def runOnlineProcessing(system,
                        EXP_NAME,
                        SCENARIOS_ROOT_DIR,
                        EXP_ROOT_DIR,
                        CACHE_ROOT_DIR,
                        SCENARIOS_SUMMARY,
                        hop_size,
                        dtw_steps=None,
                        dtw_weights=None,
                        jar_path=None,
                        use_multiprocessing=False,
                        copydir=None,
                        test_scenario=None):            # added parameter
    '''
    Runs online processing over all scenarios, clearing previous outputs first.
    '''
    logger.info(f'RunOnlineProcessing start: {EXP_NAME} | exp={EXP_ROOT_DIR} cache={CACHE_ROOT_DIR}')
    assert os.path.exists(EXP_ROOT_DIR)
    assert os.path.exists(CACHE_ROOT_DIR)
    if os.path.exists(EXP_ROOT_DIR):
        for item in os.listdir(EXP_ROOT_DIR):
            item_path = os.path.join(EXP_ROOT_DIR, item)
            if os.path.abspath(item_path) == os.path.abspath(CACHE_ROOT_DIR):
                continue
            if os.path.isdir(item_path):
                os.system(f'rm -rf "{item_path}"')
            else:
                os.remove(item_path)

    d = system_utils.get_scenario_info(SCENARIOS_SUMMARY)

    # restrict to one scenario if requested
    if test_scenario is not None:
        indices = [test_scenario - 1]
    else:
        indices = range(len(d))

    if use_multiprocessing:
        with Pool() as p:
            p.starmap(
                singleFileOnline,
                [
                    (
                        id,
                        d,
                        system,
                        EXP_NAME,
                        SCENARIOS_ROOT_DIR,
                        EXP_ROOT_DIR,
                        CACHE_ROOT_DIR,
                        hop_size,
                        dtw_steps,
                        dtw_weights,
                        jar_path,
                        copydir,
                    )
                    for id in indices
                ],
            )
    else:
        for id in indices:
            singleFileOnline(
                id,
                d,
                system,
                EXP_NAME,
                SCENARIOS_ROOT_DIR,
                EXP_ROOT_DIR,
                CACHE_ROOT_DIR,
                hop_size,
                dtw_steps,
                dtw_weights,
                jar_path,
                copydir,
            )
    logger.info(f'RunOnlineProcessing finished: {EXP_NAME} | exp={EXP_ROOT_DIR}')

def run_experiments():
    # for loop to execute all experiments
    systems = ['DTW', 'NOA', 'NOA-Mono']
    modes = ['constant']
    hop_size = 512
    
    # DTW: steps (1,1),(1,2),(2,1) with weights (2,3,3)
    dtw_steps = np.array([[1,1],[1,2],[2,1]])
    dtw_weights = np.array([2,3,3])
    
    # NOA: steps (1,1),(1,2),(2,1) with weights (1,1,2)
    noa_steps = np.array([[1,1],[1,2],[2,1]])
    noa_weights = np.array([1,1,2])
    
    # OLTW-Global: cost matrix steps (1,1),(1,0),(0,1) with weights (1,1,1) and alignment steps/weights same
    oltw_steps = np.array([[1,1],[1,0],[0,1]])
    oltw_weights = np.array([1,1,1])
    
    use_multiprocessing = False

    jar_path = "match/PerformanceMatcher.jar"
    benchmark = "train"  # train or test

    for sys_name in systems:
        for mode in modes:
            EXP_NAME = sys_name
            SCENARIOS_ROOT_DIR = f'scenarios/{benchmark}/{mode}'
            EXP_ROOT_DIR = f'experiments/{benchmark}/{mode}/{EXP_NAME}'
            CACHE_ROOT_DIR = f'{EXP_ROOT_DIR}/cache'
            SCENARIOS_SUMMARY = f'{SCENARIOS_ROOT_DIR}/scenarios.summary'
            
            # choose steps/weights based on system
            if EXP_NAME == 'DTW':
                steps_to_use = dtw_steps
                weights_to_use = dtw_weights
            elif EXP_NAME in ['NOA', 'NOA-Mono']:
                steps_to_use = noa_steps
                weights_to_use = noa_weights
            elif EXP_NAME == 'OLTW-Global':
                steps_to_use = oltw_steps
                weights_to_use = oltw_weights
            elif EXP_NAME == 'OLTW':                       # <--- new branch
                steps_to_use = None
                weights_to_use = None
            else:
                raise ValueError(f'Unknown experiment name: {EXP_NAME}')

            # select system implementation by importing the corresponding notebook
            # import_ipynb makes the notebook appear as a regular module
            if EXP_NAME == 'DTW':
                import System_NaivePairwiseDTW as system
            elif EXP_NAME in ['NOA', 'NOA-Mono']:
                import System_NOA as system
            elif EXP_NAME == 'OLTW':
                import System_OLTW as system
            elif EXP_NAME == 'OLTW-Global':
                import System_OLTW_Global as system
            else:
                raise ValueError(f'Unknown experiment name: {EXP_NAME}')

            logger.info(f'Running {EXP_NAME} on {mode} mode...')
            logger.info('Running offline processing...')
            runOfflineProcessing(
                system,
                EXP_NAME,
                SCENARIOS_ROOT_DIR,
                EXP_ROOT_DIR,
                CACHE_ROOT_DIR,
                SCENARIOS_SUMMARY,
                hop_size,
                dtw_steps=steps_to_use,
                dtw_weights=weights_to_use,
                test_scenario=None,           # pass through
            )
            logger.info('Running online processing...')
            runOnlineProcessing(
                system,
                EXP_NAME,
                SCENARIOS_ROOT_DIR,
                EXP_ROOT_DIR,
                CACHE_ROOT_DIR,
                SCENARIOS_SUMMARY,
                hop_size,
                dtw_steps=steps_to_use,
                dtw_weights=weights_to_use,
                jar_path=jar_path,
                use_multiprocessing=use_multiprocessing,
                test_scenario=None,           # pass through
            )

if __name__ == "__main__":
    run_experiments()
    logger.info('All experiments completed!')


