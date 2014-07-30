from scipy.stats import norm, rv_continuous, skew

## Skew Normal distribution
# loc = mu, scale = std, shape = skew

class skew_norm_gen(rv_continuous):
    """A skew normal continuous random variable.

    The location (loc) keyword specifies the mean.
    The scale (scale) keyword specifies the standard deviation.
    The shape (shape) keyword specifies the skewness.

    Notes
    -----
    The probability density function for `skew norm` is::

        skew_norm.pdf(x, s) = 2 * norm.pdf(x) * norm.cdf(x * s)
    """
    # default arg check is > 0
    def _argcheck(self, skew):
        return -1 < skew < 1 # Skew must be between (-1,1)
    def _pdf(self, x, skew, *args):
        return 2 * norm.pdf(x, *args) * norm.cdf(x * skew, *args)

skew_norm = skew_norm_gen(name='skew_norm', shapes='skew')
