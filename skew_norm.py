from scipy.stats import norm, rv_continuous, skew

from numpy import where, nan

import numpy.random as mtrand

## Skew Normal distribution
# loc = mu, scale = std, shape = skew

# TODO: Check this, look at weights plot

class skew_norm_gen(rv_continuous):
    """A skew normal continuous random variable.

    The location (loc) keyword specifies the mean.
    The scale (scale) keyword specifies the standard deviation.
    The shape (shape) keyword specifies the skewness.

    Notes
    -----
    The probability density function for `skew norm` is::

        skew_norm.pdf(x, s) = 2 * norm.pdf(x) * norm.cdf(x * s)


    References:
    http://azzalini.stat.unipd.it/SN/Intro/intro.html
    Python library by @author: Janwillem van Dijk
    """
    # default arg check is > 0
    def _argcheck(self, skew):
        return -1 < skew < 1 # Skew must be between (-1,1)

    def _rvs(self, skew):
        u1 = mtrand.standard_normal(self._size)
        u2 = mtrand.standard_normal(self._size)
        i = where(u2 > skew * u1)
        u1[i] *= -1.0
        return u1

    def _pdf(self, x, skew, *args):
        return 2.0  * norm.pdf(x, *args) * norm.cdf(skew * x, *args)


skew_norm = skew_norm_gen(name='skew_norm', shapes='skew')
