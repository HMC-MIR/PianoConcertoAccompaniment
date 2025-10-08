# Generates scenarios for query generation

import os
import numpy as np
from utils.query import get_query_timestamps

def myLogger(logfile, loginfo):
    '''
    Writes logging information to a specified log file.
    
    Inputs
    logfile: name of log file to generate
    loginfo: either a string or a list of strings to write to the log file
    '''
    
    assert not os.path.exists(logfile)
    with open(logfile, 'w') as f:
        if type(loginfo) == list:
            for ln in loginfo:
                f.write(ln)
        else:
            f.write(loginfo)
    return

def generateScenariosConstant(**kwargs):
    """
    Generates scenarios for query generation in constant mode.
    
    Inputs
    cnt: counter for the number of scenarios generated
    piece_id: id of the piece to generate scenarios for
    fullmix_file: file path of the full mix audio file
    tsm_factors: list of time scale modification factors to consider
    outdir: directory to save the scenarios
    QUERIES_ROOT: root directory of the queries
    ANNOT_ROOT: root directory of the annotations
    AUDIO_ROOT: root directory of the audio files
    QUERY_MEASURES_FILE: file path of the query measures file

    Returns
    cnt: updated counter for the number of scenarios generated
    logInfo: list of logging information
    """

    # parse kwargs
    cnt = kwargs['cnt']
    piece_id = kwargs['piece_id']
    fullmix_file = kwargs['fullmix_file']
    tsm_factors = kwargs['tsm_factors']
    outdir = kwargs['outdir']
    QUERIES_ROOT = kwargs['QUERIES_ROOT']
    ANNOT_ROOT = kwargs['ANNOT_ROOT']
    AUDIO_ROOT = kwargs['AUDIO_ROOT']
    QUERY_MEASURES_FILE = kwargs['QUERY_MEASURES_FILE']
    logInfo = []

    # constant mode
    for tsm_factor in tsm_factors:
        tsm_id = f'{piece_id}_P1_tsm{tsm_factor:.2f}'
        tsm_dir = f'{QUERIES_ROOT}/{piece_id}_P1/tsm{tsm_factor:.2f}'
        pref_annot_file = f'{ANNOT_ROOT}/{piece_id}_P1.beats'
        tsm_annot_file = f'{tsm_dir}/{tsm_id}_all.beats'
        o_annot_file = f'{ANNOT_ROOT}/{piece_id}_O1.beats'
        assert os.path.exists(tsm_annot_file)
        assert os.path.exists(o_annot_file)
        measures, q_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, tsm_annot_file)
        _, o_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, o_annot_file)
        _, pref_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, pref_annot_file)
        
        for (m, qt, ot, preft, queryIdx) in zip(measures, q_times, o_times, pref_times, np.arange(len(measures))+1): # queries
            cnt += 1
            scenario_dir = f'{outdir}/s{cnt}'
            os.mkdir(scenario_dir)
            cwd = os.getcwd()
            
            # original piano audio
            pref_audio = f'{cwd}/{AUDIO_ROOT}/{piece_id}_P1.wav'
            pref_link = f'{scenario_dir}/pref.wav'
            os.symlink(pref_audio, pref_link)
            
            # piano only audio (query)
            p_audio = f'{cwd}/{tsm_dir}/{tsm_id}_q{queryIdx}.wav'
            p_link = f'{scenario_dir}/p.wav' # soft links must be absolute paths
            os.symlink(p_audio, p_link)
                                
            # orchestra only audio
            o_audio = f'{cwd}/{AUDIO_ROOT}/{piece_id}_O1.wav'
            o_link = f'{scenario_dir}/o.wav'
            os.symlink(o_audio, o_link)
            
            # full mix audio
            po_audio = f'{cwd}/{AUDIO_ROOT}/{fullmix_file}'
            po_link = f'{scenario_dir}/po.wav'
            os.symlink(po_audio, po_link)
            
            # query annotation
            query_annot = f'{cwd}/{tsm_dir}/{tsm_id}_q{queryIdx}.beats'
            query_annot_link = f'{scenario_dir}/p.beats'
            os.symlink(query_annot, query_annot_link)
            
            # orchestra annotation
            o_annot = f'{cwd}/{ANNOT_ROOT}/{piece_id}_O1.beats'
            o_annot_link = f'{scenario_dir}/o.beats'
            os.symlink(o_annot, o_annot_link)
            
            # log file
            # The format is: s1 p_file o_file po_file meas_start meas_end p_start p_end o_start o_end pref_start pref_end
            logfile = f'{scenario_dir}/scenario.info'
            logstr = f's{cnt} {p_audio} {o_audio} {po_audio} {m[0]} {m[1]} {qt[0]} {qt[1]} {ot[0]} {ot[1]} {preft[0]} {preft[1]}\n'
            myLogger(logfile, logstr)
            logInfo.append(logstr)

    return {'cnt': cnt, 'logInfo': logInfo}

def generateScenariosRandom(**kwargs):
    """
    Generates scenarios for query generation in random mode.
    """
    # parse kwargs
    cnt = kwargs['cnt']
    num_queries = kwargs['num_queries']
    piece_id = kwargs['piece_id']
    fullmix_file = kwargs['fullmix_file']
    max_tsm_factors = kwargs['max_tsm_factors']
    outdir = kwargs['outdir']
    QUERIES_ROOT = kwargs['QUERIES_ROOT']
    ANNOT_ROOT = kwargs['ANNOT_ROOT']
    AUDIO_ROOT = kwargs['AUDIO_ROOT']
    QUERY_MEASURES_FILE = kwargs['QUERY_MEASURES_FILE']
    logInfo = []

    for max_tsm_factor in max_tsm_factors:
        for i in range(num_queries):
            tsm_id = f'{piece_id}_P1_tsm_random_seed{i}'
            tsm_dir = f'{QUERIES_ROOT}/{piece_id}_P1/tsm_random_max{max_tsm_factor:.2f}'
            tsm_annot_file = f'{tsm_dir}/{tsm_id}.beats'
            pref_annot_file = f'{ANNOT_ROOT}/{piece_id}_P1.beats'
            o_annot_file = f'{ANNOT_ROOT}/{piece_id}_O1.beats'
            assert os.path.exists(tsm_annot_file)
            assert os.path.exists(o_annot_file)
            measures, q_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, tsm_annot_file)
            _, o_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, o_annot_file)
            _, pref_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, pref_annot_file)
            for (m, qt, ot, preft, queryIdx) in zip(measures, q_times, o_times, pref_times, np.arange(len(measures))+1): # queries
                
                cnt += 1
                scenario_dir = f'{outdir}/s{cnt}'
                os.mkdir(scenario_dir)
                cwd = os.getcwd()
                
                # original piano audio
                pref_audio = f'{cwd}/{AUDIO_ROOT}/{piece_id}_P1.wav'
                pref_link = f'{scenario_dir}/pref.wav'
                os.symlink(pref_audio, pref_link)
                
                # piano only audio (query)
                p_audio = f'{cwd}/{tsm_dir}/{tsm_id}_q{queryIdx}.wav'
                p_link = f'{scenario_dir}/p.wav' # soft links must be absolute paths
                os.symlink(p_audio, p_link)
                                    
                # orchestra only audio
                o_audio = f'{cwd}/{AUDIO_ROOT}/{piece_id}_O1.wav'
                o_link = f'{scenario_dir}/o.wav'
                os.symlink(o_audio, o_link)
                
                # full mix audio
                po_audio = f'{cwd}/{AUDIO_ROOT}/{fullmix_file}'
                po_link = f'{scenario_dir}/po.wav'
                os.symlink(po_audio, po_link)
                
                # query annotation
                query_annot = f'{cwd}/{tsm_dir}/{tsm_id}_q{queryIdx}.beats'
                query_annot_link = f'{scenario_dir}/p.beats'
                os.symlink(query_annot, query_annot_link)
                
                # orchestra annotation
                o_annot = f'{cwd}/{ANNOT_ROOT}/{piece_id}_O1.beats'
                o_annot_link = f'{scenario_dir}/o.beats'
                os.symlink(o_annot, o_annot_link)
                
                # log file
                # The format is: s1 p_file o_file po_file meas_start meas_end p_start p_end o_start o_end pref_start pref_end
                logfile = f'{scenario_dir}/scenario.info'
                logstr = f's{cnt} {p_audio} {o_audio} {po_audio} {m[0]} {m[1]} {qt[0]} {qt[1]} {ot[0]} {ot[1]} {preft[0]} {preft[1]}\n'
                myLogger(logfile, logstr)
                logInfo.append(logstr)

    return {'cnt': cnt, 'logInfo': logInfo}

def generateScenariosContinuous(**kwargs):
    """
    Generates scenarios for query generation in continuous mode.
    """
    # parse kwargs
    cnt = kwargs['cnt']
    num_queries = kwargs['num_queries']
    piece_id = kwargs['piece_id']
    fullmix_file = kwargs['fullmix_file']
    max_alpha_changes = kwargs['max_alpha_changes']
    outdir = kwargs['outdir']
    QUERIES_ROOT = kwargs['QUERIES_ROOT']
    ANNOT_ROOT = kwargs['ANNOT_ROOT']
    AUDIO_ROOT = kwargs['AUDIO_ROOT']
    QUERY_MEASURES_FILE = kwargs['QUERY_MEASURES_FILE']
    logInfo = []

    # continuous mode
    for max_alpha_change in max_alpha_changes:
        for i in range(num_queries):
            tsm_id = f'{piece_id}_P1_tsm_continuous_seed{i}'
            tsm_dir = f'{QUERIES_ROOT}/{piece_id}_P1/tsm_continuous_max{max_alpha_change:.2f}'
            tsm_annot_file = f'{tsm_dir}/{tsm_id}.beats'
            pref_annot_file = f'{ANNOT_ROOT}/{piece_id}_P1.beats'
            o_annot_file = f'{ANNOT_ROOT}/{piece_id}_O1.beats'
            assert os.path.exists(tsm_annot_file)
            assert os.path.exists(o_annot_file)
            assert os.path.exists(pref_annot_file)
            measures, q_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, tsm_annot_file)
            _, o_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, o_annot_file)
            _, pref_times = get_query_timestamps(piece_id, QUERY_MEASURES_FILE, pref_annot_file)
            
            for (m, qt, ot, preft, queryIdx) in zip(measures, q_times, o_times, pref_times, np.arange(len(measures))+1): # queries
                
                cnt += 1
                scenario_dir = f'{outdir}/s{cnt}'
                os.mkdir(scenario_dir)
                cwd = os.getcwd()
                
                # original piano audio
                pref_audio = f'{cwd}/{AUDIO_ROOT}/{piece_id}_P1.wav'
                pref_link = f'{scenario_dir}/pref.wav'
                os.symlink(pref_audio, pref_link)
                
                # piano only audio (query)
                p_audio = f'{cwd}/{tsm_dir}/{tsm_id}_q{queryIdx}.wav'
                p_link = f'{scenario_dir}/p.wav' # soft links must be absolute paths
                os.symlink(p_audio, p_link)
                                    
                # orchestra only audio
                o_audio = f'{cwd}/{AUDIO_ROOT}/{piece_id}_O1.wav'
                o_link = f'{scenario_dir}/o.wav'
                os.symlink(o_audio, o_link)
                
                # full mix audio
                po_audio = f'{cwd}/{AUDIO_ROOT}/{fullmix_file}'
                po_link = f'{scenario_dir}/po.wav'
                os.symlink(po_audio, po_link)
                
                # query annotation
                query_annot = f'{cwd}/{tsm_dir}/{tsm_id}_q{queryIdx}.beats'
                query_annot_link = f'{scenario_dir}/p.beats'
                os.symlink(query_annot, query_annot_link)
                
                # orchestra annotation
                o_annot = f'{cwd}/{ANNOT_ROOT}/{piece_id}_O1.beats'
                o_annot_link = f'{scenario_dir}/o.beats'
                os.symlink(o_annot, o_annot_link)
                
                # log file
                # The format is: s1 p_file o_file po_file meas_start meas_end p_start p_end o_start o_end pref_start pref_end
                logfile = f'{scenario_dir}/scenario.info'
                logstr = f's{cnt} {p_audio} {o_audio} {po_audio} {m[0]} {m[1]} {qt[0]} {qt[1]} {ot[0]} {ot[1]} {preft[0]} {preft[1]}\n'
                myLogger(logfile, logstr)
                logInfo.append(logstr)

    return {'cnt': cnt, 'logInfo': logInfo}