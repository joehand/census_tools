"""
    some helpful functions for the US Census analysis
"""

import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
from scipy.stats import norm
import pylab as P
import numpy as np



def calc_params(arry):
    """ Calulates the mean and standard deviation of an array
        Also returns an linspace array for plotting density

        Args:
           arry (np Array or pd Series):  array of data

        Returns:
           x : linespace values for x
           mean : mean of array
           sigma : Standard deviation
    """
    arry = Series(arry)

    mean = arry.mean()
    variance = arry.var()
    sigma = np.sqrt(variance)
    num = arry.count()
    x = np.linspace(mean - 4.0 * sigma, mean + 4.0 * sigma, num)

    return x, mean, sigma, variance


def group_cities_norm(df, group_by='NAME', population_col='AcsTotPop',
                    sum_cols = [],
                    analysis_cols=['AcsPopDen'], min_obs=20):
    cities = []

    for name, city_df in df.groupby(group_by):
        if len(city_df[population_col]) < min_obs:
            continue

        population = city_df[population_col].sum()
        city = {
                'city' : name,
                'total_population'  : population,
            }

        city.update({column: city_df[column].sum() for column in sum_cols})

        for column in analysis_cols:
            temp_df = city_df   # city_df[city_df[column] > 0]
            weight = (temp_df[population_col] / population)
            weighted_col = (temp_df[column] * weight)
            _, mean, sigma, variance = calc_params(weighted_col)

            mean = np.log(weighted_col.sum()) #sum the column and get the mean

            """
                Is this right?
                Or should I take sum of column then get that mean?
                Right now IM doing sum of weights, which should be 1.
                That is wrong.
                What am I doing with mean there?
                Fuck.
            """

            #sigma = np.log(sigma)
            #variance = np.log(variance)

            city.update({
                    'log_' + column + '_norm_mean' : mean,
                    'log_' + column + '_norm_variance': variance,
                    'log_' + column + '_norm_sigma': sigma,
                    'sum_weight' : weight,
                })

        cities.append(city)

    cities_df = DataFrame(cities)
    return cities_df


def group_cities(df, group_by='NAME', population_col='AcsTotPop',
                    sum_cols = [],
                    analysis_cols=['AcsPopDen'], min_obs=20):
    cities = []

    for name, city_df in df.groupby(group_by):
        if len(city_df[population_col]) < min_obs:
            continue

        population = city_df[population_col].sum()
        city = {
                'city' : name,
                'total_population'  : population,
            }

        city.update({column: city_df[column].sum() for column in sum_cols})

        for column in analysis_cols:
            temp_df = city_df[city_df[column] > 0]
            log_col = np.log(temp_df[column])
            _, mean, sigma, variance = calc_params(log_col)

            city.update({
                    'log_' + column + '_mean' : mean,
                    'log_' + column + '_variance': variance,
                    'log_' + column + '_sigma': sigma
                })

        cities.append(city)

    cities_df = DataFrame(cities)
    return cities_df
