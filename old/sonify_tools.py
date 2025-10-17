# %% [markdown]
# # Sonification Tools

# %% [markdown]
# This notebook implements a number of functions that are useful in generating sonifications.

# %%
import numpy as np
import librosa as lb
import soundfile as sf
import os.path
from hmc_mir import tsm_tools
import system_utils
from multiprocessing import Pool

# %%
def get_preprocessed_wp(align_file, downsample, hop_len = None):
    '''
    Converts the warping path to seconds (if expressed in frames) and downsamples.
    
    Inputs
    align_file: filepath to the .npy file specifying the warping path between the two audio files
                If hop_len is specified, the warping path is assumed to be expressed in frames.
                If hop_len is not specified, the warping path is assumed to be expressed in seconds.
    downsample: downsample the warping path by this factor to smooth out the TSM
    hop_len: specifies the hop length in seconds between frames.  If not specified, the warping path
             is assumed to be expressed in seconds.
    
    Returns the downsampled warping path expressed in seconds.
    '''
    if hop_len is None: 
        wp = np.load(align_file) # 2xN array specifying file1-file2 alignment in sec
    else:
        wp = np.load(align_file) # 2xN array specifying file1-file2 alignment in frames
        wp = wp * hop_len # convert to sec    
    wp_middle = wp[:,1:-1] # keep ends, downsample the middle
    wp = np.hstack((wp[:,0].reshape((2,-1)), wp_middle[:,0::downsample], wp[:,-1].reshape((2,-1))))
    return wp

# %%
def mix_separate_channels(left_channel, right_channel, reweighted = True, pad = False):
    '''
    Merges two mono audio waveforms into a stereo audio waveform.  If the two waveforms differ
    in length, the longer of the two is truncated unless `pad = True`.  Provides channel volume reweighting by default.
    
    Inputs
    left_channel: the audio waveform for the left channel
    right_channel: the audio waveform for the right channel
    reweighted: if True, reweights the channels to have equal volume
    pad: if True, pads the shorter waveform with zeros to match the length of the longer waveform
    
    Returns an Nx2 array containing the mixed stereo audio waveform.
    '''
    if pad:
        if len(left_channel) > len(right_channel):
            right_channel = np.pad(right_channel, (0, len(left_channel) - len(right_channel)))
        elif len(right_channel) > len(left_channel):
            left_channel = np.pad(left_channel, (0, len(right_channel) - len(left_channel)))
        assert len(left_channel) == len(right_channel), "Channels are not the same length"
    
    N = min(len(left_channel), len(right_channel))
    mixed = np.zeros((N, 2))
    mixed[:,0] = left_channel[0:N]
    mixed[:,1] = right_channel[0:N]
    if reweighted:
        mixed = channel_volume_reweighting(mixed)
    return mixed

# %%
def channel_volume_reweighting(x_stereo):
    '''
    Reweights the left and right channels to be approximately equal volume.
    
    Inputs
    x_stereo: an Nx2 array containing the stereo audio waveform
    
    Returns an Nx2 array with the two audio channels reweighted in volume.
    '''
    mse_left = np.mean(x_stereo[:,0] * x_stereo[:,0])
    mse_right = np.mean(x_stereo[:,1] * x_stereo[:,1])
    x_stereo[:,1] = x_stereo[:,1] * np.sqrt(mse_left / mse_right)
    return x_stereo

# %%
def sonifyWithTSMSync(audiofile1, audiofile2, align_file, downsample, hop_len = None, outfile = None):
    '''
    Generates a stereo audio recording with one audio recording on the left channel and another audio
    recording on the other channel, where time-scale modification has been applied to the latter so
    that the two recordings are appropriately synchronized.  If the alignment is a subsequence alignment,
    the shorter query recording should be specified as audiofile1.
    
    Inputs
    audiofile1: filepath to the audio recording that will remain unmodified
    audiofile2: filepath to the audio recording that will be time-scaled modified
    align_file: filepath to the .npy file specifying the warping path between the two audio files.
                If hop_len is specified, the warping path is assumed to be expressed in frames.
                If hop_len is not specified, the warping path is assumed to be expressed in seconds.
    downsample: downsample the warping path by this factor to smooth out the TSM
    hop_len: specifies the hop length in seconds between frames.  If this is not specified, it will
             be assumed that the warping path is already expressed in seconds.
    outfile: the output audio file to generate
    '''
    y1, sr = lb.load(audiofile1)
    y2, sr = lb.load(audiofile2)
    if len(y1) > len(y2):
        print('Warning: If synchronization uses a subsequence alignment, the shorter query should be specified as audiofile1')
    wp = get_preprocessed_wp(align_file, downsample, hop_len) # file1-file2 alignment
    wp = system_utils.filter_vertical_and_horizontal_segments(wp)
    y2_tsm = tsm_tools.tsmvar_hybrid(y2, np.flipud(wp))
    y_mixed = mix_separate_channels(y1, y2_tsm)
    if outfile:
        sf.write(outfile, y_mixed, sr, subtype='PCM_16')
    return y_mixed

# %%
def singleSonifyWithTSMSync_batch(scenario_id, scenarios_dir, downsample, hop_len, outdir, exp_dir):
    piano_file = f'{scenarios_dir}/{scenario_id}/p.wav'
    orch_file = f'{scenarios_dir}/{scenario_id}/o.wav'
    align_file = f'{exp_dir}/{scenario_id}/hyp.npy'
    out_file = f'{outdir}/{scenario_id}.wav'
    if os.path.exists(out_file):
        # print(f'Skipping sonification of {scenario_id} -- already exists')
        return
    else:
        sonifyWithTSMSync(piano_file, orch_file, align_file, downsample, hop_len, out_file)

def sonifyWithTSMSync_batch(scenarios_dir, exp_dir, downsample, hop_len, outdir):
    '''           
    Generates stereo recordings of piano (left channel) and time-scale modified orchestra recordings
    (right channel) for all scenarios.
    
    Inputs
    scenarios_dir: directory containing all scenario directories
    exp_dir: directory containing all the hypothesis alignments
    downsample: downsample the warping path by this factor to smooth out the TSM
    hop_len: specifies the hop length in seconds between frames, needed to convert warping path to timestamps
    outdir: directory to put the generated audio files
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    summary_file = f'{scenarios_dir}/scenarios.summary'
    scenario_ids = system_utils.get_scenario_info(summary_file).keys()

    with Pool() as pool:
        pool.starmap(singleSonifyWithTSMSync_batch, [(scenario_id, scenarios_dir, downsample, hop_len, outdir, exp_dir) for scenario_id in scenario_ids])


