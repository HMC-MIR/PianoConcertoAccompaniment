import numpy as np
import librosa as lb
import soundfile as sf
from hmc_mir import tsm_tools

### Continuous Mode ###

def generate_tsm_audio_continuous(infile, outfile, max_alpha_change, seed, max_tsm_factor = 2.0):
    """
    Generates time-scale modified audio with continuous time-scale modification.
    
    Inputs
    infile: the input audio file
    outfile: the output audio file
    max_alpha_change: the maximum multiplicative alpha change from the previous frame
    seed: the seed for the random number generator
    max_tsm_factor: the maximum time-scale modification factor to use
    """
    np.random.seed(seed)
    y, sr = lb.load(infile)
    frames_in_audio = len(y) // 512 # convert to TSM analysis frames
    
    # generate alphas with continuous random walk, but clamp to [1/max_tsm_factor, max_tsm_factor]
    alphas = np.zeros(frames_in_audio)
    alphas[0] = 1
    log_max_alpha_change = np.log(max_alpha_change)
    log_max_tsm_factor = np.log(max_tsm_factor)
    log_min_tsm_factor = -log_max_tsm_factor

    log_alphas = np.zeros(frames_in_audio)
    log_alphas[0] = 0  # log(1)
    # Precompute all random increments at once for efficiency
    log_alpha_increments = np.random.uniform(-log_max_alpha_change, log_max_alpha_change, size=frames_in_audio-1)
    for i in range(1, frames_in_audio):
        log_alphas[i] = log_alphas[i-1] + log_alpha_increments[i-1]
        # Clamp to allowed range
        log_alphas[i] = np.clip(log_alphas[i], log_min_tsm_factor, log_max_tsm_factor)
    alphas = np.exp(log_alphas)
    
    alignment = np.zeros((2, frames_in_audio))
    alignment[0] = np.arange(frames_in_audio)
    alignment[1] = np.cumsum(1 / alphas)
    # shift alignment to start at 0
    alignment[1] = alignment[1] - alignment[1][0]
    # convert to seconds
    alignment *= 512 / sr
    
    # perform tsm
    y_mod = tsm_tools.tsmvar_hybrid(y, alignment, sr)
    sf.write(outfile, y_mod, sr, subtype='PCM_16')
    print("alignment done")
    return alignment