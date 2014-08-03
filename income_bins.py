import re

import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from pandas import Series
import seaborn as sns
from statsmodels import api as sm

from plot_functions import plot_single_hist, plot_ols
from utils import group_by_city

# TODO: Speed things up! Read this: http://programmers.stackexchange.com/questions/228127/is-python-suitable-for-a-statistical-modeling-application-looking-over-thousands?rq=1

def _get_values(cols):
    """ Returns array of int values based on str column names
    """
    cols = cols.values
    return [int(re.findall('\d+', col)[0]) for col in cols if re.findall('\d+', col)]


def _join_sum_cols(df, group_by='CITY_NAME', rsuffix='_CITY',
                filter='^ACSHINC([0-9])+$|^ACSHINC_COUNT$'):
    """ Returns DataFrame of same shape with columns matching filter summed by group
    """
    return df.join(
                df.groupby(group_by)[
                        df.filter(regex=filter).columns
                    ].transform(np.sum), on='ID', rsuffix=rsuffix)


def calc_inc_weights(df, group_by='CITY_NAME', weight_filter='^ACSHINC([0-9])+$',
                    sum_filter='^ACSHINC([0-9])+$|^ACSHINC_COUNT$',
                    total_count_col='ACSHINC_COUNT',
                    city_rsuffix='_CITY', weight_rsuffix='_WEIGHT'):
    """ Returns DataFrame with weights calculated for cols matching filter
    """
    if not len(df.filter(regex='_CITY').columns) > 0:
        df = _join_sum_cols(df, group_by=group_by, rsuffix=city_rsuffix, filter=sum_filter)
    cols = df.filter(regex=weight_filter).columns.values
    total_col = df[total_count_col]
    city_tot = df[total_count_col + city_rsuffix]
    for col in cols:
        data_col = df[col]
        city_col = df[col + city_rsuffix]
        df[col + weight_rsuffix] = (data_col/total_col)/(city_col/city_tot)
    return df.replace([np.inf, -np.inf], np.nan) #drop inf created from above division


def plot_inc_bins(df, filter='^ACSHINC([0-9])+$', logx=True, logy=False,
                index=None, scatter=False, legend=True,
                title=None, xlabel=None, ylabel=None):
    """ Plots income bins for specified selector

        1. Select Data
        2. Filter Columns to Bin Cols and Transpose
        3. Calculate Bin values from Columns
        4. Plot Bin Vals vs Col Counts
    """
    cols = df.filter(regex=filter).columns #save this to get x values later
    if index:
        df = df.set_index(index) #set index if we have one (useful for cities)
    df = df.filter(regex=filter).transpose()
    colors = sns.color_palette(n_colors=len(df.columns))

    for color, col in zip(colors, df.columns):
        if logx:
            x = log(_get_values(cols))
        else:
            x = _get_values(cols)

        if logy:
            data = np.log(df[col].values).dropna()
        else:
            data = df[col].values

        if index == 'CITY_NAME':
            label = col.split('-')[0]
        else:
            label = col

        if scatter:
            plt.scatter(x, data, c=color, label=str(label), s=48)
        else:
            plt.plot(x, data, c=color, label=str(label))

    if not logx:
        plt.xlim(-1, 201)

    if not xlabel:
        if logx:
            xlabel = 'Log of Income Bin Value (Log \$)'
        else:
            xlabel = 'Income Bin Value (\$)'

    if not ylabel:
        ylabel = 'Count of Households in Bin'

    if not title:
        title = 'Income Bin Counts'

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend:
        plt.legend()

    return df


def plot_incw_hist(df, filter='_WEIGHT$', kind='all', **kwargs):
    """ Return histogram of all columns in filter combined
    """
    cols = df.filter(regex=filter).columns
    if kind == 'all':
        data = Series().append([df[col] for col in cols])
    elif kind == 'bkgp':
        # sum across all l, should end up with j values
        data = df.filter(regex=filter).mean(axis=1)
    elif kind == 'income':
        # sum across all j for a city, should end up with l * # cities values
        # - group by City
        # - mean of income level weights
        # Group Data By City, calculate mean/var/stdev for 'analysis cols', add other cols
        city_df = group_by_city(df, log_analysis=False,
                                    analysis_cols = cols, **kwargs)
        data = Series().append([city_df[col] for col in city_df.filter(regex='_mean$')])
    elif kind == 'city':
        #same as income. then sum across all income means
        city_df = group_by_city(df, log_analysis=False,
                                    analysis_cols = cols, **kwargs)
        data = city_df.filter(regex='_mean$').mean(axis=1)

    return plot_single_hist(data, **kwargs)


def plot_incw_beans(df, filter='_WEIGHT$',
                        logy=True, plot_opts={}):
    # TODO: Make this color nicer. Should use seaborn colors.

    cols = df.filter(regex=filter).columns.values
    labels = [int(re.findall('\d+', col)[0]) for col in cols]
    if logy:
        data = [np.log(df[df[col] > 0][col]).dropna() for col in cols]
    else:
        data = [df[df[col] > 0][col].dropna() for col in cols]
    return sm.graphics.beanplot(data, labels=labels, plot_opts=plot_opts)

def plot_incw_pop(df, filter='_WEIGHT$', pop_col='ACSTOTPOP',
                      y_col_name = 'WEIGHT_SUM', **kwargs):
    temp_df = df
    temp_df[y_col_name] = temp_df.filter(regex=filter).mean(axis=1)

    return plot_ols(temp_df, pop_col, y_col_name, **kwargs)
