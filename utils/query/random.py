# Standard imports
import os
import pandas as pd

# External imports
import soundfile as sf
import librosa as lb
from hmc_mir import tsm_tools
import numpy as np

### Random Mode ###
def generate_alpha_random(max_tsm_factor, num_chunks, seed):
    '''Generates a random time-scale modification factor for each frame in the audio recording.'''
    np.random.seed(seed)
    min_tsm_factor = 1 / max_tsm_factor
    
    # create random time-scale modification factors for each chunk
    # uniform on log scale
    alphas_tsm_log = np.random.uniform(
        np.log(min_tsm_factor), np.log(max_tsm_factor), size=num_chunks
    )
    alphas_tsm = np.exp(alphas_tsm_log)
    return alphas_tsm # alpha value at index i is tsm factor for chunk beginning at index i and ending before index i+1

def build_alignment_random(alphas_tsm, frame_change_idx, length):
    '''Builds an alignment for the time-scale modified audio.'''
    alignment = np.zeros((2, length))
    alignment[0] = np.arange(length)
    cur_chunk = 0 # pointer to current chunk
    alphas = np.zeros(length)
    alphas[0] = alphas_tsm[0]
    for i in range(1, length):
        if i >= frame_change_idx[cur_chunk + 1] * 512:
            cur_chunk += 1 # update pointer to current chunk
        cur_alpha = alphas_tsm[cur_chunk]
        alignment[1, i] = alignment[1, i-1] + 1 / cur_alpha
        alphas[i] = cur_alpha
        
    return alignment / 22050, alphas # convert to seconds

def generate_tsm_audio_random(infile, outfile, max_tsm_factor, seed):
    '''
    Applies time-scale modification to a given audio recording and saves the generated audio to file.
    
    Inputs
    infile: The filepath of the input audio
    outfile: The filepath of the output audio
    max_tsm_factor: The maximum time-scale modification factor to apply
    seed: The seed for the random number generator
    '''
    # set random seed
    np.random.seed(seed)
    
    # load audio
    y, sr = lb.load(infile)
    frames_in_audio = len(y) // 512 # convert to TSM analysis frames
    
    # generate random number of chunks
    frame_change_idx = np.random.randint(
        low=1,
        high=frames_in_audio,
        size=np.random.randint(10, 100),
    )
    
    # sort frame change indices, remove duplicates, and add beginning and end of audio
    frame_change_idx = np.sort(frame_change_idx)
    frame_change_idx = np.unique(frame_change_idx)
    frame_change_idx = np.insert(frame_change_idx, 0, 0)
    frame_change_idx = np.append(frame_change_idx, frames_in_audio+1)
    
    # create random time-scale modification factors
    alphas_tsm = generate_alpha_random(max_tsm_factor, len(frame_change_idx), seed)
    
    # build alignment for TSM
    alignment, alphas = build_alignment_random(alphas_tsm, frame_change_idx, len(y))
    
    # apply TSM and save to file
    y_mod = tsm_tools.tsmvar_hybrid(y, alignment, sr)
    sf.write(outfile, y_mod, sr, subtype='PCM_16')
    
    # save alphas to file
    np.save(f'{outfile}.alphas.npy', alphas)
    
    return alignment

def modify_annots_tsm_random(infile, outfile, alignment):
    '''
    Modifies an annotation file according to a random time-scale modification factor.
    
    Inputs
    infile: the annotation file to be modified
    outfile: the output annotation file
    alignment: the alignment matrix for the TSM
    '''
    
    df = pd.read_csv(infile)
    
    # Extract columns
    starts = df['start'].to_numpy()
    x = alignment[0, :]   # original x-values (sorted)
    y = alignment[1, :]   # corresponding y-values

    # Interpolate df['start'] onto (x, y)
    df['start'] = np.interp(starts, x, y)
    df.to_csv(outfile, index=False)