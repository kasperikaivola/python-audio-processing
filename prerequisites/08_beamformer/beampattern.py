''' This script serves as a template for experiment 8 to
 calculate the 'Directivity patterns' using Delay-and-Sum Beamformers

 Hint : The places marked by todos should be replaced or completed by your own code
 This file is not yet executable!
'''
# This import registers the 3D projection, but is otherwise unused.
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
import os
plt.ion() # is used to prevent the figures at the end from closing

# General Parameters

script_dir = os.path.dirname(__file__)
c = 340  # Sound velocity
fs = 16000  # Sampling frequency

# Variable Parameters
n = 8  # Number of sensors

setting = 1  # Configure different simulations

if setting == 1:
    theta_0 = ...  # endfire array
    d = ...  # Distance between adjacent sensors
elif setting == 2:
    ...

theta_0_deg = theta_0/np.pi*180

dist = np.round(np.arange(n)*d, 4)  # Distances to reference microphone

bw = 200  # number of discrete angles

angles = ...  # Set of all angles (in Rad)
angles_deg = ...  # ... in Deg

bf = 256  # Number of all the discrete frequency points
bf2 = int(round(bf / 2))  # rounded for better legibility

freqs = ...  # Set of all discrete angles
#freqs = np.arange(0, fs / 2 + (fs / 2 ) * 1 / bf2, fs / 2 * 1 / bf2) # common mistake: np.arrange(0, fs/2, fs/2*1/bf2)
th = -25  # db-threshold for plots, for better graphic representation

# Array configuration
# ======================
# Calculating B(Omega)
f_n = np.zeros((bf, n))
tau = dist / c * np.cos(theta_0)  # time delay
k_n = tau * fs
for i in range(n):
    k = np.linspace(-bf2 + 1, bf2, bf)
    f_n[:, i] = np.sinc(k[:bf] - k_n[i])  # fractional delay filter

B = 1 / n * np.fft.fft(f_n, axis=0).T

# Simulation of Psi for different theta angles
# ============================================
Psi = np.zeros((bf2, bw))
Psi_db = np.zeros((bf2, bw))

for a in range(bw):  # angle index
    for f in range(bf2):  # frequency index
        # Calculating array-steering vector with dimension 1xN
        # todo: E = ?

        # Calculating directivity pattern (Matrix dimension is Bf2xBw)
        # todo: Psi[f, a] =
        # todo: Psi_db[f, a] =

        if Psi_db[f, a] < th:
            Psi_db[f, a] = th

# Hint: Loops should generally be avoided and replaced by
# matrix operations. We used loops here only to help you to understand the
# calculation procedure.

# Plot the directivity pattern
# =============================================================

# 3 dimensional representation (surface plot)
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
grid_theta, grid_f = ...  # todo: coordinate mesh grid for surface plot

plot_args = dict(rcount=bf, ccount=bw, cmap=cm.jet, linewidth=1, antialiased=True)
surf = ax1.plot_surface(grid_theta, grid_f, Psi_db, **plot_args)
# todo: set correct title and labels
...

ax1.view_init(40, 120)
plt.savefig(script_dir+'/3d_directivity_setting_'+str(setting)+'.png')
plt.show()


fig2 = plt.figure()
ax2 = fig2.gca()
plt.pcolormesh(...)  # todo plot 2D view
# todo: set correct title and labels
...
plt.savefig(script_dir+'/2d_directivity_setting_'+str(setting)+'.png')
plt.show()