import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sounddevice as sd
from hoa import *

########################################## Exercise 10.1 ##########################################
theta1 = np.deg2rad(90)
phi1   = np.deg2rad(45)

theta2 = np.deg2rad(90)
phi2   = np.deg2rad(-45)

direction1 = Direction(theta1, phi1)
direction1.plot()

direction2 = Direction(theta2, phi2)
direction2.plot()

########################################## Exercise 10.2 ##########################################
hoaSig = HoaSignal('10_hoa/scene_HOA_O4_N3D_12k.wav')
# todo code here
# sigOmni =

# Listen to signal
# How to listen
# sigNoise = np.random.randn(48000)
# sd.play(sigNoise, samplerate=48000)

########################################## Exercise 10.3 ##########################################
#todo code here

########################################## Exercise 10.4 ##########################################
#todo code here

beamformer = Beamformer.createBeamformerFromHoaSignal(hoaSig)  # Use factory method.


########################################## Exercise 10.5 ##########################################

# Steered response power map
numAzim = 160
numIncl = 80
# todo code here
# srpMap =

# Iterate over frames, calculate and plot steered response power map
frameLength = 2048
frameAdvance = 1024
nFrames = int(np.floor((hoaSig.numSamples - frameLength) / frameAdvance + 1))


def animate(i):
    sampleRange = i * frameAdvance + np.arange(frameLength)

    # Calculate steered response power map for current sample range
    srpMap.generateSrpMap(hoaSig, sampleRange)
    hplot = srpMap.updatePlot()
    hdot = srpMap.markMaximum()



fig = plt.figure()
# Don't use the scientific mode from PyCharm here!!!
anim = animation.FuncAnimation(fig, animate, frames=nFrames, init_func=srpMap.initPlot, interval=1, repeat=False)
plt.show()

########################################## Exercise 10.6 ##########################################
# Create BinauralRenderer for specified HRIR database
renderer = BinauralRenderer.createBinauralRendererFromHoaSignal(hoaSig, 'hrirs_12k.mat')
sigBinaural = renderer.renderSignal(hoaSig)

# Play binaural signal
sd.play(sigBinaural, renderer.fs)
sd.wait()
