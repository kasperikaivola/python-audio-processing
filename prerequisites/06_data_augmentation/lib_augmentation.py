import matplotlib.pyplot as plt
import numpy as np
import os
import librosa

from shutil import rmtree
from tqdm import tqdm
from scipy import signal
from sklearn.mixture import GaussianMixture


def vad_extraction(clean_speech):
    
    # YOUR CODE HERE

    return vad

def mix(clean_speech_dry, noise_dry, snr, rir):
    
    # YOUR CODE HERE

    return ...

def feature_extraction(noisy, win_length=320, hop_length=160, n_fft=512):
    # YOUR CODE BELOW
    stft = ...

    S = np.abs(stft)

    mel = ...

    mfccs = ...
    delta1 = ...
    delta2 = ...

    bandwidth = ...
    centroid = ...
    rolloff = ...

    zero_crossing_rate = ...

    features = np.vstack((mfccs, delta1, delta2, bandwidth, centroid, rolloff, zero_crossing_rate)).astype(np.float32)

    return features

class GenSpeech:
    def __init__(self, path):
        pass # YOUR CODE INSTEAD

    def __next__(self):
        pass # YOUR CODE INSTEAD

    def __iter__(self):
        pass # YOUR CODE INSTEAD

    def __len__(self):
        pass # YOUR CODE INSTEAD

def create_gen_rir(path):
    pass # YOUR CODE INSTEAD


def get_noise(path, len_desired):
    """ Returns a random noise signal. If the noise file is longer than len_desired a random excerpt
        is returned. If the noise file is shorter than len_desired it is looped.

        Parameters:
            path (str): path to folder containing noise files
                as wavs
            len_desired (int): Length of the noise signal to be returned

        Returns:
            np.array: noise signal
    """

    files = [file for file in os.listdir(path) if file[-4:] == '.wav']
    success = False
    while not success:
        it_file = np.random.randint(len(files))
        path_file = os.path.join(path, files[it_file])
        wav, fs = librosa.load(path_file, sr=16E3)

        # first repeat wav if too short
        if len(wav) < len_desired:
            num_reps = int(np.ceil(len_desired / len(wav)))
            wav = np.tile(wav, num_reps)

        # now truncate wav if too long
        if len(wav) > len_desired:
            idx_start = np.random.randint(len(wav) - len_desired)
            wav = wav[idx_start:idx_start + len_desired]

        if np.sqrt(np.mean(np.square(wav))) > 1E-12:
            success = True

    return wav


def run_augmentation(path, debug=False):
    """ Contains the main loop for data "Augmentation" """

    # Input Paths
    path_clean_speech = os.path.join(path, 'clean-speech')
    path_noise = os.path.join(path, 'noise')
    path_rirs  = os.path.join(path, 'rirs')

    # Output Path
    path_output = os.path.join(path, 'output')

    if not debug:
        if os.path.exists(path_output):
            rmtree(path_output)
        os.mkdir(path_output)

    # Generators
    gen_clean_speech = GenSpeech(path_clean_speech)
    gen_rirs = create_gen_rir(path_rirs)
    gen_rand = np.random.default_rng(seed=42)

    # Main Loop
    for idx, clean in tqdm(enumerate(gen_clean_speech), total=len(gen_clean_speech)):
        # for debugging
        if debug and (idx > 100):
            break

        # YOUR CODE HERE

        # save output
        np.savez(
            os.path.join(path_output, str(idx) + '.npz'), 
            features=features, 
            vad=vad
        )