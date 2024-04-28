import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


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
        # x_block =

        # Filtering the input speech signal using room impulse response and adaptive filter. Please note that you don't
        # need to implement the complete filtering here. A simple vector manipulation would be enough here
        # todo your code:
        # x_tilde[k] =
        # x_hat[k] =

        # Calculating the estimated error signal
        # todo your code
        # err[k] =

        # Updating the filter
        # todo your code
        # h =
        # Calculating the absolute system distance
        # todo your code
        # s_diff[k] =

        k = k + 1  # time index

    # Calculating the relative system distance in dB
    # todo your code
    # s_diff = 10 * np.log10(s_diff[:k] /  HERE! ).T

    return s_diff, err, x_hat, x_tilde


# switch between exercises
exercise = 1  # choose between 1-7

# load data
f = np.load('04_echocomp/echocomp.npz')
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
    # x[1] =        #white noise

    # x[2] =        #colorful noise

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

    fig, axs = plt.subplots(3)
    # todo your code for ex. 1
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
