from tempfile import tempdir
import matplotlib.pyplot as plt
import numpy as np
import os

# General Parameters:
script_dir = os.path.dirname(__file__)
c = 340  # Sound velocity (m/s)
fs = 16e3  # Sampling frequency
n = 8  # Number of sensors

# Simulation parameters and settings
# ==========================================
setting = 1  # Configure different simulations

if setting == 1:
    theta_0 = ...  # endfire array
    d = ...  # Distance between adjacent sensors
    title = ...
elif setting == 2:
    ...
    
# dist = np.arange(n)*d  # last value is 0.6000001 instead of 0.6???
dist = np.round(np.arange(n)*d, 4)  # Distances to reference microphone

bf = 256  # Number of all the discrete frequency points
bf2 = int(round(bf / 2))  # rounded for better legibility

freqs = np.linspace(0, fs / 2, bf2)  # Set of all discrete angles
#freqs = np.arange(0, fs / 2 + (fs / 2 ) * 1 / bf2, fs / 2 * 1 / bf2) # common mistake: np.arrange(0, fs/2, fs/2*1/bf2)
th = -25  # db-threshold for plots, for better graphic representation

# Array configuration
# ======================
# Calculating B(Omega)
f_n = np.zeros((bf, n))
tau = ... # todo
k_n = ... # todo
for i in range(n):
    k = np.linspace(-bf2 + 1, bf2, bf)
    f_n[:, i] = np.sinc(k[:bf] - k_n[i])  # fractional delay filter

B = 1 / n * np.fft.fft(f_n, axis=0).T

# Calculating Psi(Omega,theta_0)
Psi = np.zeros(bf2)
E = np.zeros((n,bf2))

for f in range(bf2): # frequency index
    # Calculating array-steering vector with dimension 1xN
    E = ... # todo

    # Calculating directivity pattern for target direction only (Matrix dimension is Bf2x1)
    Psi[f] = ...

Bconj = np.conj(B).T

Psi_diff = np.zeros(bf2)
Psi_wn = np.zeros(bf2)
arg = (2 * freqs * d / c).T

for i in range(n):
    Psi_wn = ...
    for m in range(n):
        Psi_diff = ...

G = 10 * np.log10( ... )  # Calculating Array Gain
Gwn = 10* np.log10( ... )  # Calculating White Noise Gain

# optional but interesting
f_aliasing = c/d*1/(1+np.cos(theta_0))

# Plots
# =============================================================
# Think of a representation that enables you to do the comparisons of all settings as asked in the exercises.
fig, axs = plt.subplots(2)

axs[0].plot(freqs, G)
axs[0].title.set_text('Array Gain ('+title+', d = ' + str(d) + 'cm), Aliasing above ' + str(round(f_aliasing)) + 'Hz')
axs[0].set_xlabel('Frequency [Hz]')
axs[0].set_ylabel('Gain [dB]')
# think of meaningful y limits
axs[0].grid(True)

axs[1].plot(freqs, Gwn)
axs[1].title.set_text('White Noise Gain ('+title+', d = ' + str(d) + 'cm)')
axs[1].set_xlabel('Frequency [Hz]')
axs[1].set_ylabel('Gain [dB]')
axs[1].grid(True)
# think of meaningful y limits
plt.savefig(script_dir+'/arraygain_setting_'+str(setting)+'.png')
plt.show()