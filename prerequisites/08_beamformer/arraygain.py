from tempfile import tempdir
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
np.seterr(invalid='raise')

# General Parameters:
script_dir = os.path.dirname(__file__)
c = 340  # Sound velocity (m/s)
fs = 16e3  # Sampling frequency
n = 8  # Number of sensors

# Simulation parameters and settings
# ==========================================
setting = 1  # choose between 1-2
d = 0.0425
try:
    #print(sys.argv[1])
    setting = int(sys.argv[1])
    d = float(sys.argv[2])
except:
    pass

if setting == 1:
    theta_0 = 0 # endfire array (0 degrees)
    #d = ...  # Distance between adjacent sensors
    title = ' Endfire Array'
elif setting == 2:
    theta_0 = np.pi / 2  # broadside array (90 degrees)
    #d = 4.25 / 100  # Distance between adjacent sensors (converted to meters)
    title = ' Broadside Array'
    
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
tau = dist / c * np.cos(theta_0)  # time delay
k_n = tau * fs # todo
for i in range(n):
    k = np.linspace(-bf2 + 1, bf2, bf)
    f_n[:, i] = np.sinc(k[:bf] - k_n[i])  # fractional delay filter

B = 1 / n * np.fft.fft(f_n, axis=0).T

# Calculating Psi(Omega,theta_0)
Psi = np.zeros(bf2,dtype=complex)
Psi_db = np.zeros(bf2,dtype=complex)
E = np.zeros((n,bf2),dtype=complex)

#for f in range(bf2): # frequency index
    # Calculating array-steering vector with dimension 1xN
    #E = np.exp(-1j * 2 * np.pi * freqs[f] * dist / c * np.cos(theta_0)) # todo
#    E[:, f] = np.exp(1j * 2 * np.pi * freqs[f] * tau)  # Array steering vector

    # Calculating directivity pattern for target direction only (Matrix dimension is Bf2x1)
    #Psi[f] = np.abs(np.dot(B[:, f], E.T))
#    Psi[f] = np.abs(np.dot(B[:, f], E[:, f].T))

for f in range(bf2):  # frequency index
    # Calculating array-steering vector with dimension 1xN
    # todo: E = ?
    E = np.exp(1j * 2 * np.pi * freqs[f] * dist / c * np.cos(theta_0))

    # Calculating directivity pattern (Matrix dimension is Bf2xBw)
    # todo: Psi[f, a] =
    #print(B.shape)
    #print(E.shape)
    Psi[f] = np.abs(np.dot(B[:,f], E.T))**2
    #Psi[f] = np.abs(np.sum(B[:,f], E.T))
    # todo: Psi_db[f, a] =
    #Psi_db[f] = 20 * np.log10(Psi[f])

    #if Psi_db[f] < th:
    #    Psi_db[f] = th

Bconj = np.conj(B).T
print("B.shape ",B.shape)
print("bf2: ",bf2)
Psi_diff = np.zeros(bf2,dtype=complex) # Psi(omega,theta) difference, theta=constant
Psi_wn = np.zeros(bf2,dtype=complex) # Psi(omega,theta) white noise, theta=constant
print("Psi_diff.shape: ",Psi_diff.shape)
print("Psi_wn.shape: ",Psi_wn.shape)
arg = (2 * freqs * d / c).T

print(B.shape,Bconj.shape,Psi_diff.shape,arg.shape)
#go through 8mics
for i in range(n): #n=number of mic
    #Psi_wn = ... #todo
    #Psi_wn += np.abs(B[i, :bf2]) ** 2  # White noise gain calculation
    Psi_wn += np.abs(B[i,:bf2]) ** 2  # White noise gain calculation
    for m in range(n): #n=number of mic
        #Psi_diff = ... #todo
        temp = B[i,:bf2] * Bconj[:bf2,m]
        #print(temp)
        Psi_diff += np.sinc(arg * (i - m))*temp  # Difference gain calculation

G = 10 * np.log10( Psi / Psi_diff )  # Calculating Array Gain
Gwn = 10* np.log10( Psi / Psi_wn )  # Calculating White Noise Gain

# optional but interesting
f_aliasing = (c/d)*(1/(1+np.cos(theta_0)))

# Plots
# =============================================================
# Think of a representation that enables you to do the comparisons of all settings as asked in the exercises.
fig, axs = plt.subplots(2)
plt.subplots_adjust(hspace=0.8)
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
plt.savefig(script_dir+'/arraygain_setting_'+str(setting)+'_'+str(d)+'.png')
#plt.show()
plt.close()