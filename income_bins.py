import re

import matplotlib.pyplot as plt
import numpy as np
from numpy import log
from pandas import Series
import seaborn as sns
from statsmodels import api as sm

from plot_functions import plot_single_hist, plot_ols


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
    for col in cols:
        df[col + weight_rsuffix] = (
                df[col]/df[total_count_col])/(
                df[col + city_rsuffix]/df[total_count_col + city_rsuffix])
    return df.replace([np.inf, -np.inf], np.nan).dropna(how="all") #drop inf created from above division


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
            data = df[col]
            data = np.log(data)
            data = data.dropna()
        else:
            data = df[col]

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
            xlabel = 'Log of Income Bin Value (Log $)'
        else:
            xlabel = 'Income Bin Value ($)'

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


def plot_incw_hist(df, filter='_WEIGHT$',
                    log=True, normalize=False,
                    dist_names=[], skew_fit=False,
                    bins = 50,
                    title=None, xlabel=None, ylabel=None):
    """ Return histogram of all columns in filter combined
    """
    cols = df.filter(regex=filter).columns
    data = Series().append([df[col] for col in cols])
    return plot_single_hist(data, log=log, normalize=normalize,
                        dist_names=dist_names, skew_fit=skew_fit,
                        bins=bins,
                        title=title, xlabel=xlabel, ylabel=ylabel)


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
