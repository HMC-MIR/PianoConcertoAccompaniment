import system_utils
import os
import librosa as lb
import numpy as np
from scipy.io.wavfile import write

""""
This module contains functions for offline processing. This will contain funcitons that
compute features, then saving them to their respective folders.
features saved in features/
"""

def getCacheDir(d, scenario_id):
    '''Inputs:
    d: dictionary that contains all the scenario infos
    scenario_id: the id for a scenario, for ex: s1, s2 and so on
    Output: cache_id -- the piece id of the specific scenario'''
    o_id = os.path.splitext(os.path.basename(d[scenario_id]['o']))[0] # e.g. rach2_mov1_O1
    p_id = '_'.join(o_id.split('_')[:-1]) + '_P1' # e.g. rach2_mov1_P1
    cache_id = p_id # e.g. rach2_mov1_P1
    return cache_id

def compute_features(scenario_root, feature_mode='chroma_stft_norm2', hop_length=512):
    '''
    Inputs:
    scenario_root -- directory to the scenario root dir, scenarios/{benchmark}/{mode[mode_id]}
    feature_mode -- only implemented for chroma_stft_norm2 now
    hop_length -- hop size, default is 512. 

    Outputs: It will save the following:
    1. p_ref features under features/{piece_id}/{feature_mode}/
    2. p_query features in their respective scenario folders
    '''
    cache_root = f'features'
    os.makedirs(cache_root, exist_ok=True)

    scenario_summary_dir = f'{scenario_root}/scenarios.summary'
    d = system_utils.get_scenario_info(scenario_summary_dir, input_type="summary")

    print(f'Processing {len(d)} scenarios in {scenario_root}')

    for i in range(len(d)):
        s_id = f's{i+1}'
        scenario_dir = f'{scenario_root}/{s_id}'

        #1. Determine the piece name
        system_utils.verify_scenario_dir(scenario_dir) #scenarios/{benchmark}/mode/s{s_id}
        piece_id = getCacheDir(d, s_id)

        # 2. Setup directory: features/{piece_id}/{feature_mode}/
        pref_cache_dir = os.path.join(cache_root, piece_id, feature_mode)
        os.makedirs(pref_cache_dir, exist_ok=True)

        # 3. Define save path
        save_pref_dir = f'{pref_cache_dir}/{piece_id}.features.npy'
        save_pquery_dir = f'{scenario_dir}/pquery_stft.npy'
        # Check if the file already exists to avoid redundant computation

        if not os.path.exists(save_pref_dir):
            pref_file = f'{scenario_dir}/pref.wav'
            if os.path.exists(pref_file):
                print(f'[{s_id}] Computing p_ref for {piece_id}...')
                y_pref, _ = lb.load(pref_file)
                F_pref = lb.feature.chroma_stft(y=y_pref, 
                                                sr=22050, 
                                                hop_length=hop_length, 
                                                center=False, 
                                                norm=2)
                np.save(save_pref_dir, F_pref)
        else:
            print(f'[{s_id}] p_ref already exists in cache.')
        if not os.path.exists(save_pquery_dir):
            pquery_file = f'{scenario_dir}/p.wav'
            if os.path.exists(pquery_file):
                print(f'[{s_id}] Computing p_query for {piece_id}...')
                y_pquery, _ = lb.load(pquery_file)
                F_pquery = lb.feature.chroma_stft(y=y_pquery, 
                                                sr=22050, 
                                                hop_length=hop_length, 
                                                center=False, 
                                                norm=2)
                np.save(save_pquery_dir, F_pquery)
            else:
                print(f'[{s_id}] Warning: p.wav not found')
        else:
            print(f'[{s_id}] p_query already exists.')

def oltw_offline_processing(scenario_root, hop_length=512):
    '''
    Specific to OLTW where we only need to chop the p_ref corresponding to when the scenarios
    start and end.
    Inputs:
    scenario_root -- directory to the scenario root dir, scenarios/{benchmark}/{mode[mode_id]}
    hop_length -- hop size, default is 512. 

    Output: Saves an audio file, pref_chopped.wav in its respective scenario folder.
    '''
    scenario_summary_dir = f'{scenario_root}/scenarios.summary'
    d = system_utils.get_scenario_info(scenario_summary_dir, input_type="summary")

    print(f'Processing {len(d)} scenarios in {scenario_root}')

    for i in range(len(d)):
        s_id = f's{i+1}'
        scenario_dir = f'{scenario_root}/{s_id}'

        # Verify scenario directory
        system_utils.verify_scenario_dir(scenario_dir)
        
        # Get piano reference boundaries
        pref_path = os.path.join(scenario_dir, "pref.wav")
        if not os.path.exists(pref_path):
            raise FileNotFoundError(f"pref.wav missing in {scenario_dir}")
        
        try:
            p_start_t, p_end_t = system_utils.get_piano_reference_boundaries(scenario_dir)
        except (AssertionError, FileNotFoundError, ValueError) as e:
            raise ValueError(f"Cannot find piano reference boundaries in scenario.info: {e}")
        
        if not os.path.exists(os.path.join(scenario_dir, "pref_chopped.wav")):
            # Load and write chopped file
            y_ref, sr_ref = lb.load(pref_path, sr=None)
            y_chopped = y_ref[int(p_start_t * sr_ref): int(p_end_t * sr_ref)]
            
            out_path = os.path.join(scenario_dir, "pref_chopped.wav")
            write(out_path, sr_ref, (y_chopped * 32767).astype("int16"))  # 16-bit PCM
        else:
            print(f'[{s_id}] pref_chopped.wav exists already, ready for OLTW alignment.')
        

def offline_processing(system, modes=None, benchmark='train'):
    '''
    Overall pipeline, where the processes are done specific to the system
    system -- str, includes 'DTW', 'NOA', 'NOA-MONOTONIC', 'OLTW' and 'OLTW-GLOBAL'
    modes -- an array of strings, but default checks constant, random, and continuous modes.
    benchmark -- str, can be 'train' or 'test'
    '''
    if modes == None:
        modes = ['constant', 'random', 'continuous']
        for mode in modes:
            scenario_root = os.path.join("scenarios", benchmark, mode)
            if system == 'DTW' or system == 'NOA' or system == 'NOA-MONOTONIC' or system == 'OLTW-GLOBAL':
                compute_features(scenario_root)
            elif system == 'OLTW':
                oltw_offline_processing(scenario_root)
    return
