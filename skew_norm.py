import scipy.stats as stats

## Skew Normal distribution
# loc = mu, scale = std, shape = skew

_norm_pdf_C = np.sqrt(2*pi)
_norm_pdf_logC = np.log(_norm_pdf_C)

def _norm_pdf(x):
    return exp(-x**2/2.0) / _norm_pdf_C




class skew_norm(norm):
    """A skew normal continuous random variable.

    The location (loc) keyword specifies the mean.
    The scale (scale) keyword specifies the standard deviation.
    The shape (shape) keyword specifies the skewness.

    Notes
    -----
    The probability density function for `skew norm` is::

        skew_norm.pdf(x, shape) = 2 * norm.pdf(x) * norm.cdf(x * shape)
    """

    def fit(self, data, **kwds):
        """

        """
        floc = kwds.get('floc', None)
        fscale = kwds.get('fscale', None)
        fshape = kwds.get('fshape', None)

        if floc is not None and fscale is not None and fshape is not None:
            # This check is for consistency with `rv_continuous.fit`.
            # Without this check, this function would just return the
            # parameters that were given.
            raise ValueError("All parameters fixed. There is nothing to "
                             "optimize.")

        data = np.asarray(data)

        if floc is None:
            loc = data.mean()
        else:
            loc = floc

        if fscale is None:
            scale = np.sqrt(((data - loc)**2).mean())
        else:
            scale = fscale

        if fshape is None:
            shape =
        else:
            shape = fshape

        return loc, scale, shape


def _skewnormal_parms(mean=0.0, stdev=1.0, skew=0.0):
    if abs(skew) > Plot._skew_max():
        #print('Skewness must be between %.8f and %.8f' % (
        #                                        -Plot._skew_max(), Plot._skew_max()))
        return None, None, None
        """
            if we want to show all values regardless of bounds, use this
        skew = copysign(Plot._skew_max(), skew)
        """


    beta = (2.0 - pi / 2.0)
    skew_23 = pow(skew * skew, 1.0 / 3.0)
    beta_23 = pow(beta * beta, 1.0 / 3.0)
    eps2 = skew_23 / (skew_23 + beta_23)
    eps = copysign(sqrt(eps2), skew)
    delta = eps * sqrt(pi / 2.0)
    alpha = delta / sqrt(1.0 - delta * delta)
    omega = stdev / sqrt(1.0 - eps * eps)
    xi = mean - omega * eps
    return xi, omega, alpha

def _skew_max():
    beta = 2.0 - pi / 2.0
    #lim(delta, shape-> inf) = 1.0
    eps = sqrt(2.0 / pi)
    return beta * pow(eps, 3.0) / pow(1.0 - eps * eps, 3.0 / 2.0) - 1e-16

def _pdf_skewnormal(x, location=0.0, scale=1.0, shape=0.0, normalize=True):
    if not normalize:
        return 2.0 * norm.pdf(x * norm.cdf(shape * x))
    if location and scale:
        t = (x - location) / scale
        return 2.0 / scale * norm.pdf(t) * norm.cdf(shape * t)
    else:
        return None

def calc_skew_pdf(self, data):
    xpdf, mean, stdev = self._calc_params(data)
    num = data.count()
    skew = stats.skew(data)
    locm, scalem, shapem = self._skewnormal_parms(mean, stdev, skew)
    xpdf = np.linspace(mean - 4.0 * stdev, mean + 4.0 * stdev, num)
    y_min = self._pdf_skewnormal(xpdf, locm, scalem, shapem)
    #print 'Skew: %s, Mean %s, StDev: %s' % (str(skew), str(mean), str(stdev))
    return xpdf, y_min, skew
