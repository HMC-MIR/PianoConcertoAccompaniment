# Standard imports
import shutil
import pandas as pd

# External imports
import soundfile as sf
import librosa as lb
from hmc_mir import tsm_tools

### Constant Mode ###
def generate_tsm_audio_constant(infile, outfile, tsm_factor):
    '''
    Applies time-scale modification to a given audio recording and saves the generated audio to file.
    
    Inputs
    infile: The filepath of the input audio
    outfile: The filepath of the output audio
    tsm_factor: The time-scale modification factor to apply
    '''
    if tsm_factor == 1: # just copy the file
        shutil.copyfile(infile, outfile)
    else:
        y, sr = lb.load(infile)
        y_mod = tsm_tools.tsm_hybrid(y, tsm_factor, sr)
        sf.write(outfile, y_mod, sr, subtype = 'PCM_16')
        
def modify_annots_tsm_constant(infile, outfile, tsm_factor):
    '''
    Modifies an annotation file according to a single global time-scale modification factor.
    
    Inputs
    infile: the annotation file to be modified
    tsm_factor: the time-scale modification factor to apply
    outfile: the output annotation file
    '''
    df = pd.read_csv(infile)
    df['start'] = df['start'] * tsm_factor
    df.to_csv(outfile, index=False)