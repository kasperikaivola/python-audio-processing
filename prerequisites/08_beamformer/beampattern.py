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
import sys
plt.ion() # is used to prevent the figures at the end from closing

# General Parameters

script_dir = os.path.dirname(__file__)
c = 340  # Sound velocity
fs = 16000  # Sampling frequency

# Variable Parameters
n = 8  # Number of sensors

#setting = 1  # Configure different simulations

# switch between exercises
setting = 1  # choose between 1-2
d = 4.25/100
try:
    #print(sys.argv[1])
    setting = int(sys.argv[1])
    d = float(sys.argv[2])
except:
    pass

if setting == 1:
    theta_0 = 0  # endfire array (180 degrees)
    #d = 4.25 / 100  # Distance between adjacent sensors
    title = ' Endfire Array'
elif setting == 2:
    theta_0 = np.pi / 2  # broadside array (90 degrees)
    #d = 4.25 / 100  # Distance between adjacent sensors (converted to meters)
    title = ' Broadside Array'

theta_0_deg = theta_0/np.pi*180

dist = np.round(np.arange(n)*d, 4)  # Distances to reference microphone

bw = 200  # number of discrete angles

angles = np.linspace(-np.pi, np.pi, bw) # Set of all angles (in Rad)
angles_deg = angles / np.pi * 180  # ... in Deg

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
        E = np.exp(1j * 2 * np.pi * freqs[f] * dist / c * np.cos(angles[a]))

        # Calculating directivity pattern (Matrix dimension is Bf2xBw)
        # todo: Psi[f, a] =
        Psi[f, a] = np.abs(np.dot(B[:, f], E.T))
        # todo: Psi_db[f, a] =
        Psi_db[f, a] = 20 * np.log10(Psi[f, a])

        if Psi_db[f, a] < th:
            Psi_db[f, a] = th

# Hint: Loops should generally be avoided and replaced by
# matrix operations. We used loops here only to help you to understand the
# calculation procedure.

# Plot the directivity pattern
# =============================================================

# 3 dimensional representation (surface plot)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
grid_theta, grid_f = np.meshgrid(angles_deg, freqs)  # todo: coordinate mesh grid for surface plot

plot_args = dict(rcount=bf, ccount=bw, cmap=cm.jet, linewidth=1, antialiased=True)
surf = ax1.plot_surface(grid_theta, grid_f, Psi_db, **plot_args)
ax1.view_init(30, 60)
# todo: set correct title and labels
ax1.set_title('3D Directivity Pattern, setting '+str(setting)+', '+str(title)+'sensor dist d='+str(d))
ax1.set_xlabel('Angle (degrees)')
ax1.set_xlim([-150,150])
ax1.set_ylabel('Frequency (Hz)')
ax1.set_ylim([0,8000])
ax1.set_zlabel('Gain (dB)')
ax1.set_zlim([-25,0])

#ax1.view_init(40, 120)
plt.savefig(script_dir+'/3d_directivity_setting_'+str(setting)+'sensor dist d='+str(d)+'.png')
#plt.show()

fig2 = plt.figure()
ax2 = fig2.gca()
pcol = plt.pcolormesh(grid_theta, grid_f, Psi_db, shading='auto', cmap=cm.jet)  # todo plot 2D view
fig2.colorbar(pcol, ax=ax2)
# todo: set correct title and labels
ax2.set_title('2D Directivity Pattern, setting '+str(setting)+', '+str(title)+'sensor dist d='+str(d))
ax2.set_xlabel('Angle (degrees)')
ax2.set_ylabel('Frequency (Hz)')
plt.savefig(script_dir+'/2d_directivity_setting_'+str(setting)+'sensor dist d='+str(d)+'.png')
#plt.show()
plt.close('all')