import matplotlib.pyplot as plt
import numpy as np
import os
import librosa

from shutil import rmtree
from tqdm import tqdm
from scipy import signal
from sklearn.mixture import GaussianMixture

from os import listdir
from os.path import isfile, join, abspath
import scipy.signal
import random

def vad_extraction(clean_speech):
    # Compute RMS energy for each frame
    rms = librosa.feature.rms(y=clean_speech, frame_length=320, hop_length=160)[0]

    # Compute log power
    log_power = 10 * np.log10(rms**2)

    # Fit a GMM with two components to the speech power
    gmm = GaussianMixture(n_components=2, random_state=0).fit(log_power.reshape(-1, 1))

    # Identify which cluster corresponds to speech (typically the cluster with higher mean log power)
    speech_cluster = np.argmax(gmm.means_)

    # Compute the posterior probabilities for each frame
    posterior_probs = gmm.predict_proba(log_power.reshape(-1, 1))[:, speech_cluster]

    # Apply a threshold to obtain the VAD decision (e.g., 0.5)
    vad = posterior_probs > 0.5

    return vad

def mix(clean_speech_dry, noise_dry, snr, rir):
    #snr specified in db scale
    # YOUR CODE HERE
    #filter the clean speech and noise signals with their respective RIRs
    clean_speech_filtered = scipy.signal.fftconvolve(clean_speech_dry, rir[0], mode='full')[:len(clean_speech_dry)]
    noise_filtered = scipy.signal.fftconvolve(noise_dry, rir[1], mode='full')[:len(noise_dry)]
    
    #scale the noise to achieve the specified SNR
    clean_speech_power = np.mean(clean_speech_filtered**2)
    noise_power = np.mean(noise_filtered**2)
    snr_linear = 10**(snr / 10)
    scaling_factor = np.sqrt(clean_speech_power / (snr_linear * noise_power))
    scaled_noise = scaling_factor * noise_filtered

    #sum the clean speech and scaled noise
    noisy_speech = clean_speech_filtered + scaled_noise

    #normalize the resulting signal if its absolute value exceeds one
    max_abs_noisy_speech = np.max(np.abs(noisy_speech))
    if max_abs_noisy_speech > 1:
        normalization_factor = 1 / max_abs_noisy_speech
        noisy_speech = noisy_speech * normalization_factor
        clean_speech_filtered = clean_speech_filtered * normalization_factor

    return clean_speech_filtered, noisy_speech

def feature_extraction(noisy, win_length=320, hop_length=160, n_fft=512):
    # YOUR CODE BELOW
    stft = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hamming')

    S = np.abs(stft)

    mel = librosa.feature.melspectrogram(S=S**2) #melspectrogram takes S**2 as input

    #first 40 MFCCs
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=40)
    #first 12 frequency bins of the MFCC's first order delta features
    delta1 = librosa.feature.delta(mfccs, order=1)[:12, :]
    #first 6 frequency bins of the MFCC's second order delta features
    delta2 = librosa.feature.delta(mfccs, order=2)[:6, :]

    #spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S)
    #spectral centroid
    centroid = librosa.feature.spectral_centroid(S=S)
    #spectral roll-off
    rolloff = librosa.feature.spectral_rolloff(S=S)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(noisy, frame_length=win_length, hop_length=hop_length)
    #each row is a feature, convert to single precision
    features = np.vstack((mfccs, delta1, delta2, bandwidth, centroid, rolloff, zero_crossing_rate)).astype(np.float32)

    return features

class GenSpeech:
    def __init__(self, path):
        # YOUR CODE INSTEAD
        self.path = abspath(path)
        self.files = listdir(path)

    def __next__(self):
        # YOUR CODE INSTEAD
        if len(self.files) == 0:
            raise StopIteration
        file = self.files.pop()
        speech, _ = librosa.load(join(self.path, file), sr=16E3)
        return speech

    def __iter__(self):
        # YOUR CODE INSTEAD
        return self

    def __len__(self):
        # YOUR CODE INSTEAD
        return len(self.files)

def create_gen_rir(path):
    # YOUR CODE INSTEAD
    rir_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    
    while True:
        rir_file = random.choice(rir_files)
        filepath = os.path.join(path, rir_file)
        rir, _ = librosa.load(filepath, sr=16E3)
        yield rir

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