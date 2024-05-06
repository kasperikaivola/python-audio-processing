import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import os
import sys

def nlms4echokomp(x, g, noise, alpha, lh):
    """ The Python function 'nlms4echokomp' simulates a system for acoustic echo compensation using NLMS algorithm
    :param x:       Input speech signal from far speaker
    :param g:       impluse response of the simulated room
    :param noise:   Speech signal from the near speaker and the background noise(s + n)
    :param alpha:   Step size for the NLMS algorithm
    :param lh:      Length of the compensation filter

    :return s_diff:  relative system distance in dB
    :return err:    error signal e(k)
    :return x_hat:  output signal of the compensation filter
    :return x_tilde:acoustic echo of far speakers
    """

    # Initialization of all the variables
    lx = len(x)  # Length of the input sequence
    lg = len(g)  # Length of the room impulse response(RIR)
    if lh > lg:
        lh = lg
        import warnings
        warnings.warn('The compensation filter is shortened to fit the length of RIR!', UserWarning)

    # Vectors are initialized to zero vectors.
    x_tilde = np.zeros(lx - lg)
    x_hat = x_tilde.copy()
    err = x_tilde.copy()
    s_diff = x_tilde.copy()
    h = np.zeros(lh)

    # Realization of NLMS algorithm
    k = 0
    for index in range(lg, lx):
        # Extract the last lg values(including the current value) from the
        # input speech signal x, where x(i) represents the current value.
        # todo your code
        x_block = x[index-lg:index]

        # Filtering the input speech signal using room impulse response and adaptive filter. Please note that you don't
        # need to implement the complete filtering here. A simple vector manipulation would be enough here
        # todo your code:
        # x_tilde[k] =
        # x_hat[k] =
        x_tilde[k] = np.dot(h, x_block)  # Convolution
        x_hat[k] = x_tilde[k] - noise[index-lg]  # Removing the noise component

        # Calculating the estimated error signal
        # todo your code
        # err[k] =
        err[k] = x[index] - x_hat[k]

        # Updating the filter
        # todo your code
        # h =
        h += alpha * err[k] * x_block / (np.linalg.norm(x_block)**2)
        # Calculating the absolute system distance
        # todo your code
        # s_diff[k] =
        s_diff[k] = np.abs(x[index] - x_tilde[k])

        k = k + 1  # time index

    # Calculating the relative system distance in dB
    # todo your code
    # s_diff = 10 * np.log10(s_diff[:k] /  HERE! ).T
    s_diff = 10 * np.log10(s_diff / np.mean(x ** 2))

    return s_diff, err, x_hat, x_tilde


# switch between exercises
exercise = 1  # choose between 1-7
try:
    #print(sys.argv[1])
    exercise = int(sys.argv[1])
except:
    pass

# load data
#print(os.path.abspath("."))
f = np.load('./echocomp.npz')
g = [f['g1'], f['g2'], f['g3']] # three different room impulse responses
s = f['s'] # speech
fs = f['fs']

# declare variables
ls = len(s)  # length of the speech signal
vn = 3  # number of curves

# generation of default values
alpha = 0.1  # step size for NLMS
n0 = np.sqrt(0.16) * np.random.randn(ls)  # white noise
s = s / np.sqrt(s.T.dot(s)) * np.sqrt(n0.T.dot(n0))  # normalize speech signal to power 0.16

# input variables for nlms4echokomp
noise = [np.zeros(ls,) for i in range(vn)]  # zero noise array (no disturbance by noise)
alphas = [alpha for i in range(vn)]  # step size factor for different exercises
lh = len(g[0]) * np.ones(vn, dtype=int)  # length of the compensation filter
x = [n0.copy() for i in range(vn)]  # white noise as input signal

# In the following part, the matrices and vectors must be adjusted to
# meet the requirements for the different exercises. Note that for exercise 2-6 
# you must use g[0]. Do this without changing the for loop at the end.
# (exercise 1 can be simulated using only the initialized values above)

if exercise == 2:
    # Only the value of input speech signal need to be changed. All the other
    # vectors and parameters should not be modified

    x[0] = s  # Speech signal
    # todo your code
    x[1] = np.random.normal(scale=np.sqrt(0.16), size=np.size(s))  #white noise
    b = [1]  # numerator coefficient vector in a 1-D sequence
    a = [1, -0.5]  # denominator coefficient vector in a 1-D sequence 
    x[2] = lfilter(b, a, x[1])  #colorful noise
    #use a single impulse response for all 3 cases
    g[1]=g[0]
    g[2]=g[0]


    #s_diff_speech, _, _, _ = nlms4echokomp(x[0], g[0], np.zeros(ls), alpha, 200)
    #s_diff_rw, _, _, _ = nlms4echokomp(x[1], g[0], np.zeros(ls), alpha, 200)
    #s_diff_rc, _, _, _ = nlms4echokomp(x[2], g[0], np.zeros(ls), alpha, 200)

    leg = ('Speech', 'white noise', 'colorful noise')
    title = 'Different Input Signals'

elif exercise == 3:
    # todo your code
    # noise[0] =
    # noise[1] =
    # noise[2] =
    # leg =
    # title =
    pass
elif exercise == 4:
    # consider, which input variables of nlms4echokomp() you have to change
    # do it similar as in the previous elif section
    # todo your code
    pass

elif exercise == 5:
    # todo your code
    pass

elif exercise == 6:
    # todo your code
    pass

elif exercise == 7:
    # todo your code
    pass

# There should be appropriate legends and axis labels in each figure!
if exercise == 1:
    s_diff, e, x_h, x_t = nlms4echokomp(n0, g[0], np.zeros(ls), alpha, 200)

    #calculate ERLE in dB
    ERLE_dB = 10 * np.log10(np.square(x_t) / np.square(e))

    fig, axs = plt.subplots(3)
    # todo your code for ex. 1
     #plot echo signal and residual echo
    axs[0].plot(x_t, label='Echo Signal', alpha=0.7)
    axs[0].plot(e, label='Residual Echo', alpha=0.7)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    
    #plot relative system distance
    axs[1].plot(s_diff, label='Relative System Distance (D(k))')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()
    
    #plot ERLE measure in dB
    axs[2].plot(ERLE_dB, label='ERLE (dB)')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Magnitude (dB)')
    axs[2].legend()
    plt.show()
else:
    for i in range(vn):
        # 3 system distances with different parameters are calculated here
        # The input variables of 'nlms4echokomp' must be adapted according
        # to different exercises.

        s_diff, e, x_h, x_t = nlms4echokomp(x[i], g[i], noise[i], alphas[i], lh[i])
        plt.plot(s_diff, label=leg[i])

    plt.title('Exercise ' + str(exercise) + ': ' + title)
    plt.xlabel('k')
    plt.ylabel('D(k) [dB]')
    plt.grid(True)
    plt.legend()
    plt.show()