import numpy as np
import librosa as lb
import soundfile as sf
import pandas as pd
import re

def get_audio_files(audio_summary_file, regexp):
        '''
        Returns a list of audio filenames matching a given regular expression.
        
        Inputs
        regexp: a string specifying the regular expression
        '''
        
        df = pd.read_csv(audio_summary_file)
        p_list = [a for a in df['id'] if re.search(regexp, a)] 
        return p_list

def get_query_timestamps(piece_id, query_measures_file, annot_file):
        '''
        This function infers the timestamp locations of all queries in a piano only recording.
        
        Inputs
        piece_id: A string specifying the piece and movement, e.g. 'rach2_mov1'
        query_measures_file: Filepath to the query.measures file that specifies the measures in each query
        annot_file: Filepath to the annotation file that specifies timestamps for measure downbeats
        
        Returns a list of (tstart,tend) tuples that indicate the starting and ending timestamps 
        of each query in the piano only recording.
        '''

        # read annotation file
        df = pd.read_csv(annot_file) # has two columns: start (timestamp) and measure (number)
        
        # get query measure info
        d = {}
        with open(query_measures_file,'r') as f:
            for line in f: 
                parts = line.split(',') # e.g. 'rach2_mov1,1-75,83-161,177-297,313-374'
                cur_piece = parts[0]
                parts.pop(0)
                d[cur_piece] = parts
        if piece_id not in d:
            raise Exception(f"Cannot find entry for {piece_id} in {query_measures_file}.  Aborting.")
            
        # infer timestamps        
        times = []
        measures = []
        for pair in d[piece_id]:
            parts = pair.split('-')
            assert len(parts) == 2
            start_measure, end_measure = parts
            start_time = float(df.loc[df['measure'] == int(start_measure), 'start'])
            end_time = float(df.loc[df['measure'] == int(end_measure), 'start'])
            measures.append((int(start_measure), int(end_measure)))
            times.append((start_time, end_time))
                        
        return measures, times
    
def extract_audio_excerpt(infile, outfile, starttime, endtime):
    '''
    Extracts an audio segment from a given audio recording and writes the output to file.
    
    Inputs
    infile: The input audio recording from which the excerpt should be taken
    outfile: The output audio file to write
    starttime: The start time in seconds of the selected segment
    endtime: The end time in seconds of the selected segment
    '''
    y, sr = lb.load(infile)
    start_sample = int(np.round(starttime * sr))
    end_sample = int(np.around(endtime * sr))
    assert end_sample < len(y)
    sf.write(outfile, y[start_sample:end_sample], sr, subtype='PCM_16')
    
def modify_annots_select(infile, outfile, select_start, select_end):
    '''
    Modifies an annotation file by selecting a specified interval in the recording.
    Only annotations that fall within the interval will be included in the modified 
    annotation file, and the timestamps will be expressed relative to the interval start time.
    
    Inputs
    infile: the annotation file to be modified
    outfile: the output annotation file
    select_start: the start of the selected interval (in sec)
    select_end: the end of the selected interval (in sec)
    '''
    df = pd.read_csv(infile)
    select_rows = (df['start'] >= select_start) & (df['start'] <= select_end)
    df.loc[:,'start'] = df['start'] - select_start
    df = df[select_rows]
    df.to_csv(outfile, index=False)