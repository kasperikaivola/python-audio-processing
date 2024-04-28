import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm.notebook import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import PredefinedSplit

from librosa.segment import agglomerative
from librosa import load, stft
import librosa.feature as feature
import sounddevice as sd
from scipy.signal import convolve as conv

# Parameters

stft_params = dict(win_length=512, n_fft=2048, hop_length=256, window='hann')
frames_select = 5
rel_thresh = 0.5
min_dr = 40
min_length = 0.3
"""     
        stft_params: parameter dictionary that is passed to librosa.stft
        frames_select: number of segments per utterance
        min_dr: examples whose dynamic range is lower are discarded
        min_length: examples that are shorter (after trim) are discarded
        rel_thresh: threshold to trim the signal [0 - 1]
"""

def load_dataset(path, digits=None):
    """
    Loads digits from the command data set and extracts features

    Parameters:
        path: File path where the folders per digit are located
        digits: Which digits to load, e.g. ['four', 'two', 'zero', 'seven']. If None, all digits are loaded

    Returns:
        x_train: training data
        t_train: training labels
        x_test: test data
        t_test: test_labels
        val_split: PredefinedSplit to separate training and validation data

    """
    x_train = []
    t_train = []
    x_test = []
    t_test = []
    val_fold = []

    all_digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    str2num = {string: num for num, string in enumerate(all_digits)}
    if digits is None:
        digits = all_digits


    test_list = open(os.path.join(path, 'testing_list.txt')).read().split('\n')
    val_list = open(os.path.join(path, 'validation_list.txt')).read().split('\n')

    for digit_num, digit_str in enumerate(digits):
        path_digit = os.path.join(path, digit_str)
        files = os.listdir(path_digit)
        for file in tqdm(files, desc=digit_str, dynamic_ncols=True, unit='files'):
            wav, fs = load(os.path.join(path_digit, file), sr=None)

            wav = trim(wav, fs)
            if type(wav) is int:
                continue

            instance = wav2instance(wav, fs)

            if digit_str + '/' + file in test_list:
                x_test.append(instance)
                t_test.append(str2num[digit_str])
            elif digit_str + '/' + file in val_list:
                x_train.append(instance)
                t_train.append(str2num[digit_str])
                val_fold.append(0)
            else:
                x_train.append(instance)
                t_train.append(str2num[digit_str])
                val_fold.append(-1)
    val_split = PredefinedSplit(val_fold)
    return x_train, t_train, x_test, t_test, val_split


def _movmean(x, window=None):
    """Compute the moving median over x, needed in trim"""
    if window is None:
        window = 320
    if type(window) is int:
        window = np.ones((window, ))
    window /= np.sum(window)

    return conv(x, window, mode='same')


def _movstd(x, window=None):
    """Compute the moving standard deviation over x, needed in trim"""
    if window is None:
        window = 320
    if type(window) is int:
        window = np.ones((window, )) / window
    window /= np.sum(window)

    return np.sqrt(_movmean(np.square(x - _movmean(x, window)), window))


def trim(wav, fs, check_dr=True):
    """removes pauses in wav using a threshold criterion

       Params:
       rel thresh (global): relative threshold where 0 is the minimum and 1 is the maximum
       level in wav
       min_dr (global): the sample is discarded if the dynamic range is smaller.
       min_length (global): the sample is discarded if the selected part is shorter
       check_dr: Wether to check the dynamic range criterion. Should be false for testing and true for training data.

       Returns:
           Truncated Sample, or -1 if the sample was discarded

    """
    # power level
    lvl = _movmean(20 * np.log10(_movstd(wav, 320) + 1E-12), 1024)
    dr = np.max(lvl) - np.min(lvl)

    if (dr < min_dr) & check_dr:
        return -1

    thresh = np.min(lvl) + rel_thresh * dr
    wav = wav[lvl > thresh]

    if float(len(wav))/fs < min_length:
        return -2

    return wav


def _reduce(X, idx):
    """Compute median over feature segments, whose borders are specified by idx"""
    ll = [np.median(X[..., idx[i]:idx[i + 1]], axis=-1) for i in range(len(idx) - 1)]
    return np.stack(ll, axis=1)


def wav2instance(wav, fs):
    """Converts the audio sequence 'wav' to a feature dictionary"""

    # normalize
    wav -= np.mean(wav)
    wav /= np.sqrt(np.mean(np.square(wav)))

    instance = dict()

    S = stft(wav, **stft_params)

    log_mel = np.log(feature.melspectrogram(S=np.abs(S)**2, sr=fs, fmax=8000))
    instance['log_mel'] = log_mel

    mfcc = feature.mfcc(S=log_mel, n_mfcc=30)
    instance['mfcc'] = mfcc

    # solution to optional task 8:


    idx_keep = agglomerative(instance['log_mel'], frames_select + 2)
    idx_keep = idx_keep[1:]
    for key in instance:
        instance[key] = _reduce(instance[key], idx_keep)
    return instance


def test_utter(estimator):
    """Records an audio snippet, extracts features and presents them to an instance"""
    fs = 16000  # Sample rate
    print("Start")
    x = sd.rec(int(2 * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Stop")

    _, ax = plt.subplots(1, 3)
    ax[0].plot(x)
    ax[0].set_title("Recording")

    x = trim(x[:, 0], fs, check_dr=False)
    if type(x) is int:
        if x == -2:
            print("Could not Extract Sequence")
            return


    ax[1].plot(x)
    ax[1].set_title("Trimmed")
    inst = wav2instance(x, fs)

    try:
        proba = estimator.predict_proba([inst])

        ax[2].stem(estimator.classes_, proba[0])
        ax[2].set_xticks(estimator.classes_)
        ax[2].set_title("Posteriors")

        cls = estimator.classes_[np.argmax(proba)]
    except AttributeError:
        cls = estimator.predict([inst])

    print("Predicted: {}".format(cls))


def analyze_grid_search(clf):
    for param_name in clf.param_grid:
        vals = clf.param_grid[param_name]
        if len(vals) < 2:
            continue
        mean_score = []
        x_plot = []
        xx = np.array([])
        yy = np.array([])
        xticklabels = []
        for x, val in enumerate(vals):
            idx = clf.cv_results_['param_' + param_name] == val
            x_plot.append(x)
            mean_score.append(np.mean(clf.cv_results_['mean_test_score'][idx]))
            xx = np.concatenate((xx, x * np.ones(np.count_nonzero(idx))))
            yy = np.concatenate((yy, clf.cv_results_['mean_test_score'][idx]))
            if type(val) is not str:
                if abs(np.log10(val)) < 2:
                    val = ("{:.2f}".format(val))
                else:
                    val = ("{:.2e}".format(val))
            xticklabels.append(val)

        fig, ax = plt.subplots()
        fig.set_figwidth(min(10, len(xticklabels)))
        ax.set_title(param_name)
        ax.scatter(xx, yy, alpha=0.8)
        ax.plot(x_plot, mean_score)

        if len(xticklabels) > 15:
            xticks = np.floor(np.linspace(0, len(xticklabels), 10)).astype(int)
            xticklabels = xticklabels(idx)
        else:
            xticks = range(len(vals))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        plt.show()


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
        Selects only certain features from the input and transforms them to stacked and flattened vectors. For instance
        mfcc=0, log_mel='all' discards the mfccs but takes all log_mels. When a number n is specified, the first n
        features are selected per frame. For instance mfcc=10 selects the first 10 mfccs per frame and stacks them to
        one vector per instance

        Parameters:
            mfcc:       number of mfccs per frame. 0 to discard, 'all' for all
            log_mel:    number of log mel bands per frame. 0 to discard, 'all' for all
    """
    def __init__(self, mfcc=0, log_mel=0):
        self.mfcc = mfcc
        self.log_mel = log_mel
        # solution to ex. 7:

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X2 = []
        for idx, data in enumerate(X):
            temp = np.array([])
            for key, val in self.get_params().items():
                if val == 'all':
                    sel = data[key][:, :]
                else:
                    sel = data[key][:val, :]

                if temp.shape[0] == 0:
                    temp = sel
                else:
                    temp = np.vstack((temp, sel))
            X2.append(temp)

            # flatten
        return np.concatenate([x.reshape((1, -1)) for x in X2], axis=0)
