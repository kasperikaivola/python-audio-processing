"""https://ieeexplore.ieee.org/document/1395954"""

from scipy.stats import rv_continuous
from scipy.special import gamma as gfun
from scipy.interpolate import interp1d
import numpy as np


class GammaGen(rv_continuous):
    """Generalized two sided gamma distribution

        Arguments:
            eta [0.5]
            beta [1]
            gamma [1]

    """
    def __init__(self):
        super().__init__(self)
        xp = np.arange(-1, 1, 1/1000)
        eta = 0.97
        beta = 40
        gamma = 0.99
        pdf = ((gamma * beta ** eta) / (2*gfun(eta))) * (np.abs(xp) ** (eta*gamma - 1))  \
               * np.exp(-1*beta * np.abs(xp)**gamma)
        cdf = np.nancumsum(pdf)
        cdf /= max(cdf)
        self._interpcdf = interp1d(xp, cdf, kind='linear', assume_sorted=True, bounds_error=False, fill_value=(0, 1))
        self._interpppf = interp1d(cdf, xp, kind='linear', assume_sorted=True, bounds_error=False, fill_value=(-1, 1))

    def cdf(self, x, *args, **kwds):
        return self._interpcdf(x)

    def ppf(self, q, *args, **kwds):
        return self._interpppf(x)

def randg(size):
    """draw samples from a generalized two sided gamma distribution

        Arguments:
            size: number of samples

    """
    gen = GammaGen()
    return gen.rvs(size=size)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = randg(12000)
    plt.hist(x, bins=np.arange(-0.25, 0.25, 1/500), histtype='step', density=True)
    plt.show()
