# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 16:19:12 2013

@author: Janwillem van Dijk
@email: jwe.van.dijk@xs4all.nl

Module for generating skew normal random numbers (Adelchi Azzalini)
===================================================================
http://azzalini.stat.unipd.it/SN/

Licensing:
This code is distributed under the GNU LGPL license.

-   rnd_skewnormal: returns random valuse for sn distribution with given
        location scale and shape
-   random_skewnormal: returns random valuse for sn distribution with given
        mean, stdev and skewness
-   skewnormal_parms: returns location, scale and shape given
        mean, stdev and skewnessof the sn distribution
-   skewnormal_stats: returns mean, stdev and skewness given
        location scale and shape
-   pdf_skewnormal: returns values for the pdf of a skew normal distribution
-   cdf_skewnormal: returns values for the cdf of a skew normal distribution
-   T_owen returns: values for Owens T as used by cdf_skewnormal
-   skew_max: returns the maximum skewness of a sn distribution
"""
from math import sqrt, copysign, pi
import numpy.random as random
from numpy import where, zeros, ones, float64, array
from numpy import linspace, median, arange, take, sort
from numpy import inner, kron
from numpy import exp as np_exp
from numpy import arctan as np_arctan
from scipy.stats import norm
from scipy.special import gamma as sp_gamma

try:
    """
    Try to use owen.f90 compiled into python module with
    f2py -c -m owens owens.f90
    ginving owens.so
    http://people.sc.fsu.edu/~jburkardt/f_src/owens/owens.f90
    """
    owens = None
    #import owens
except:
    print 'owens not found'


def T_Owen_int(h, a, jmax=50, cut_point=6):
    """
    Return Owens T
    ==============
    @param: h   the h parameter of Owen's T
    @param: a   the a parameter of Owen's T (-1 <= a <= 1)
    Python-numpy-scipy version for Owen's T translated from matlab version
        T_owen.m of R module sn.T_int
    """
    if type(h) in (float, float64):
        h = array([h])
    low = where(h <= cut_point)[0]
    high = where(h > cut_point)[0]
    n_low = low.size
    n_high = high.size
    irange = arange(0, jmax)
    series = zeros(h.size)
    if n_low > 0:
        h_low = h[low].reshape(n_low, 1)
        b = fui(h_low, irange)
        cumb = b.cumsum(axis=1)
        b1 = np_exp(-0.5 * h_low ** 2) * cumb
        matr = ones((jmax, n_low)) - b1.transpose()  # matlab ' means transpose
        jk = kron(ones(jmax), [1.0, -1.0])
        jk = jk[0: jmax] / (2 * irange + 1)
        matr = inner((jk.reshape(jmax, 1) * matr).transpose(),
                     a ** (2 * irange + 1))
        series[low] = (np_arctan(a) - matr.flatten(1)) / (2 * pi)
    if n_high > 0:
        h_high = h[high]
        atana = np_arctan(a)
        series[high] = (atana * np_exp(-0.5 * (h_high ** 2) * a / atana) *
                    (1.0 + 0.00868 * (h_high ** 4) * a ** 4) / (2.0 * pi))
    return series


def fui(h, i):
    return (h ** (2 * i)) / ((2 ** i) * sp_gamma(i + 1))


def T_Owen_series(h, a, jmax=50, cut_point=6):
    """
    Return Owens T
    ==============
    @param: h   the h parameter of Owen's T
    @param: a   the a parameter of Owen's T
    Python-numpy-scipy version for Owen's T
    Python-numpy-scipy version for Owen's T translated from matlab version
        T_owen.m of R module sn.T_Owen
    """
    if abs(a) <= 1.0:
        return T_Owen_int(h, a, jmax=jmax, cut_point=cut_point)
    else:
        """D.B. Owen Ann. Math. Stat. Vol 27, #4 (1956), 1075-1090
         eqn 2.3, 2.4 and 2.5
         Available at: http://projecteuclid.org/DPubS/Repository/1.0/
            Disseminate?view=body&id=pdf_1&handle=euclid.aoms/1177728074"""
        signt = copysign(1.0, a)
        a = abs(a)
        h = abs(h)
        ha = a * h
        gh = norm.cdf(h)
        gha = norm.cdf(ha)
        t = 0.5 * gh + 0.5 * gha - gh * gha - \
                T_Owen_int(ha, 1.0 / a, jmax=jmax, cut_point=cut_point)
        return signt * t


def T_Owen(h, a):
    """
    Return Owens T
    ==============
    @param: h   the h parameter of Owens T
    @param: a   the a parameter of Owens T
    Try to use owens.f90 version else python version
    owens.f90 is approximately a factor 100 faster
    """
    if owens:
        """Owen's T using owens.f90 by Patefield and Brown
            http://www.jstatsoft.org/v05/a05/paper
            Fortran source by Burkhard
            http://people.sc.fsu.edu/~jburkardt/f_src/owens/owens.f90"""
        if type(h) in [float, float64]:
            return owens.t(h, a)
        else:
            t = zeros(h.size)
            for i in range(h.size):
                t[i] = owens.t(h[i], a)
            return t
    else:
        """
        Owens T after sn.T_Owen(H, a) D.B. Owen (1956)
        """
        return T_Owen_series(h, a)


def cdf_skewnormal(x, location=0.0, scale=1.0, shape=0.0):
    """
    Return skew normal cdf
    ======================
    @param location:    location of sn distribution
    @param scale:       scale of sn distribution
    @param shape:       shape of sn distribution
    http://azzalini.stat.unipd.it/SN/
    """
    xi = (x - location) / scale
    return norm.cdf(xi) - 2.0 * T_Owen(xi, shape)


def pdf_skewnormal(x, location=0.0, scale=1.0, shape=0.0):
    """
    Return skew normal pdf
    ======================
    @param location:    location of sn distribution
    @param scale:       scale of sn distribution
    @param shape:       shape of sn distribution
    http://azzalini.stat.unipd.it/SN/
    """
    t = (x - location) / scale
    return 2.0 / scale * norm.pdf(t) * norm.cdf(shape * t)


def rnd_skewnormal(location=0.0, scale=1.0, shape=0.0, size=1):
    """
    Return skew normal random values
    ================================
    with given location, scale and shape
    @param location:    location of sn distribution
    @param scale:       scale of sn distribution
    @param shape:       shape of sn distribution
    @param size:        number of values to generate
    http://azzalini.stat.unipd.it/SN/ Matlab source rsn.m in sn-matlab.zip
    """
    u1 = random.normal(0.0, 1.0, size=size)
    u2 = random.normal(0.0, 1.0, size=size)
    i = where(u2 > shape * u1)
    u1[i] *= -1.0
    return location + scale * u1


def skewnormal_parms(mean=0.0, stdev=1.0, skew=0.0):
    """
    Return parameters for a skew normal distribution function
    =========================================================
    @param mean:    mean of sn distribution
    @param stdev:   standard deviation of sn distribution
    @param skew:    skewness of sn distribution
    http://azzalini.stat.unipd.it/SN/Intro/intro.html
        location (xi), scale (omega) and shape (alpha)
    """
    if abs(skew) > skew_max():
        print('Skewness must be between %.8f and %.8f' % (
                                                -skew_max(), skew_max()))
        print('None, None, None returned')
        return None, None, None
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


def skewnormal_stats(location=0.0, scale=1.0, shape=0.0):
    """
    Return statistics of a skew normal distribution function
    ========================================================
    @param location:    location of sn distribution
    @param scale:       scale of sn distribution
    @param shape:       shape of sn distribution
    http://azzalini.stat.unipd.it/SN/Intro/intro.html
    """
    beta = 2.0 - pi / 2.0
    delta = shape / sqrt(1.0 + shape * shape)
    eps = delta * sqrt(2.0 / pi)
    mean = location + scale * eps
    stdev = scale * sqrt(1.0 - eps * eps)
    skew = beta * pow(eps, 3.0) / pow(1.0 - eps * eps, 3.0 / 2.0)
    return mean, stdev, skew


def skew_max():
    """
    Return maximum skewness of a skew normal distribution
    =====================================================
    skewness for shape --> infinity
    """
    beta = 2.0 - pi / 2.0
    #lim(delta, shape-> inf) = 1.0
    eps = sqrt(2.0 / pi)
    return beta * pow(eps, 3.0) / pow(1.0 - eps * eps, 3.0 / 2.0) - 1e-16


def random_skewnormal(mean=0.0, stdev=1.0, skew=0.0, size=1):
    """
    Return random numbers from a skew normal distribution
    =====================================================
    with given mean, stdev and shape
    @param mean:    mean of sn distribution
    @param stdev:   stdev of sn distribution
    @param shape:   shape of sn distribution
    @param shape:   shape of sn distribution
    """
    loc, scale, shape = skewnormal_parms(mean, stdev, skew)
    if loc is not None:
        return rnd_skewnormal(loc, scale, shape, size=size)
    else:
        return None
