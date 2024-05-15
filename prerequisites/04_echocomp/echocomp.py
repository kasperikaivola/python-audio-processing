import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter


def nlms4echokomp(x, g, noise, alpha, lh):
    """ The Python function 'nlms4echokomp' simulates a system for acoustic echo compensation using NLMS algorithm
    :param x:       Input speech signal from far speaker
    :param g:       impulse response of the simulated room
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

    # Ensure h is zero-padded if lh < lg
    if lh < lg:
        h = np.zeros(lg)  # Zero pad h to the length of g
    else:
        h = np.zeros(lh)

    # Initialize vectors to zero
    x_tilde = np.zeros(lx - lg)
    x_hat = x_tilde.copy()
    err = x_tilde.copy()
    s_diff = x_tilde.copy()

    # Realization of NLMS algorithm
    k = 0
    for index in range(lg, lx):
        # Extract the last lg values(including the current value) from the
        # input speech signal x, where x(i) represents the current value.
        x_block = x[index - lg:index]

        # Filtering the input speech signal using room impulse response and adaptive filter
        x_tilde[k] = np.dot(g, x_block)  # Using full g

        # Noise is added here
        x_tilde[k] += noise[k]

        # Only use the relevant part of h if lh < lg
        x_hat[k] = np.dot(h[:lg] if lh < lg else h, x_block)

        # Calculating the estimated error signal
        err[k] = x_tilde[k] - x_hat[k]

        # Updating the filter
        # Updating the filter
        power = np.linalg.norm(x_block) ** 2
        if power != 0:
            update = alpha * err[k] * x_block / power
            h[:lh] += update[:lh]  # Ensure the update is only applied within the bounds of h

        # Calculating the absolute system distance
        diff = np.linalg.norm(g[:lh] - h[:lh]) ** 2  # Consider only the first lh coefficients
        s_diff[k] = diff / np.linalg.norm(g[:lh]) ** 2

        k += 1

    # Calculating the relative system distance in dB
    s_diff = 10 * np.log10(s_diff[:k])

    return s_diff, err, x_hat, x_tilde


# switch between exercises
exercise = 6  # choose between 1-7

# load data
f = np.load('echocomp.npz') #04_echocomp/
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

    x[0] = s  # Speech signal s
    # todo your code
    r_w = np.random.normal(scale=np.sqrt(0.16),size=np.size(s))
    x[1] = r_w        #white noise r_w

    b = [1]           #numerator b
    a = [1,-0.5]      #dominator a
    r_c = lfilter(b,a,x[1])
    x[2] = r_c        #colorful noise r_c

    #use a single impulse response for all 3 cases
    g[1]=g[0]
    g[2]=g[0]

    leg = ('Speech', 'white noise', 'colorful noise')
    title = 'Different Input Signals'
elif exercise == 3:
    # todo your code
    noise[0] = np.random.normal(scale=np.sqrt(0.0),size=np.size(s))
    noise[1] = np.random.normal(scale=np.sqrt(0.001),size = np.size(s))
    noise[2] = np.random.normal(scale=np.sqrt(0.01),size = np.size(s))
    #use a single impulse response for all 3 cases
    g[1]=g[0]
    g[2]=g[0]


    leg = ['No Noise ($\sigma^2=0$)', 'Low Noise ($\sigma^2=0.001$)', 'High Noise ($\sigma^2=0.01$)']
    title ='Different background noise'
    pass
elif exercise == 4:
    # consider, which input variables of nlms4echokomp() you have to change
    # do it similar as in the previous elif section
    # todo your code
    noise[0] = np.random.normal(scale=np.sqrt(0), size=np.size(s))
    noise[1] = np.random.normal(scale=np.sqrt(0.001), size=np.size(s))
    noise[2] = np.random.normal(scale=np.sqrt(0.01), size=np.size(s))
    # speech as input signal in all cases
    x[0]=s
    x[1]=s
    x[2]=s
    #same impulse response for all
    g[1]=g[0]
    g[2]=g[0]
    leg = ('σ^2=0','σ^2=0.001','σ^2=0.01')
    title = 'Different background noise levels, input=speech'
    pass

elif exercise == 5:
    # todo your code
    #white noise, variance=0.01
    noise[0:3] = [np.random.normal(scale=np.sqrt(0.01), size=np.size(s)) for _ in range(3)]

    #excitation signal: white noise, var=0.16
    #stepsize alpha: {0.1,0.5,1.0}
    alphas[0]=0.1
    alphas[1]=0.5
    alphas[2]=1.0
    #same impulse response for all
    g[1]=g[0]
    g[2]=g[0]
    leg = ('alpha=0.1','alpha=0.5','alpha=1.0')
    title = 'Different stepsizes (alpha), input=white noise, var=0.16'
    pass

elif exercise == 6:
    # todo your code

    #lh[0] = len(g[0]) - 10
    #lh[1] = len(g[0]) - 30
    #lh[2] = len(g[0]) - 60

    # same impulse response for all
    #g[1] = g[0]
    #g[2] = g[0]

    # Defining different lengths for the compensation filter for each variant
    lh_lengths = [len(g[0]) - diff for diff in [10, 30, 60]]

    # Update the lh values for each setup
    lh[0], lh[1], lh[2] = lh_lengths

    # Keep using the same impulse response g[0] for all variants
    g[1] = g[0]
    g[2] = g[0]

    # Labels for plot legends to reflect different filter lengths
    leg = ['lh=' + str(lh_lengths[0]), 'lh=' + str(lh_lengths[1]), 'lh=' + str(lh_lengths[2])]
    title = 'Different length of transversal filter (lh)'


    pass

elif exercise == 7:
    # todo your code

    # Adjust the length of compensation filters to match each corresponding impulse response
    lh = [len(g[i]) for i in range(vn)]  # Ensure lh matches the exact lengths of g[0], g[1], g[2]

    # No background noise is present
    noise = [np.zeros(ls) for _ in range(vn)]

    # Step size is the same for all cases since it is not specified to vary
    alphas = [alpha] * vn

    # Legends and title are adjusted for clarity
    leg = ['g1 Length = ' + str(len(g[0])), 'g2 Length = ' + str(len(g[1])), 'g3 Length = ' + str(len(g[2]))]
    title = 'Room impulse responses of different lengths'

    pass

# There should be appropriate legends and axis labels in each figure!
if exercise == 1:
    s_diff, e, x_h, x_t = nlms4echokomp(n0, g[0], np.zeros(ls), alpha, 200)
    ERLE_dB = 10 * np.log10((np.square(x_t)) / np.square(e))

    fig, axs = plt.subplots(3)
    # todo your code for ex. 1
    #echo and residual
    axs[0].plot(x_t, label='Echo Signal $\\tilde{x}(k)$')
    axs[0].plot(e, label='Residual Signal $e(k)$')
    axs[0].set_title('Echo Signal and Residual Signal ')
    axs[0].legend(fontsize='small',handlelength=2, handletextpad=0.5,loc='upper right',borderpad=0.2,labelspacing=0.1,borderaxespad=0.1)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    #relative system distance in dB
    axs[1].plot(s_diff)
    axs[1].set_title('Relative System Distance $D(k)$ in dB')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('dB')

    #ERLE measure
    axs[2].plot(ERLE_dB)
    axs[2].set_title('ERLE Measure $e(k)$')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('dB')

    plt.tight_layout()
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




