from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import scipy.stats as stats
from scipy.stats import norm

from .dist_fits import fit_dist, calculate_BIC
from .skew_norm import skew_norm
from .skew_normal import pdf_skewnormal, skewnormal_parms
from .utils import ols_reg

def plot_single_hist(data, ax=None,
                    logx=False, normalize=False,
                    dist_names=[], skew_fit=False,
                    bins=20, print_bic=True, xlim=None,
                    title=None, xlabel=None, ylabel='Probability'):
    data = data.dropna()
    obs = len(data)

    if logx:
        data = np.log(data[data > 0])

    if normalize:
        data = (data - data.mean())/data.std()

    if ax is None:
        ax = plt.gca()

    # Histogram of the data
    # TODO: Add cumulative option, cumulative = True; need to do it for dist_fits too
    ax.hist(data.values, bins, normed=True, histtype="stepfilled", alpha=0.7)

    if len(dist_names) or skew_fit:
        xmin, xmax = data.min(), data.max()
        x = np.linspace(xmin, xmax, obs) # Create x vals

    for dist_name in dist_names:
        param, BIC = fit_dist(data, dist_name)
        dist = getattr(stats, dist_name)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
        dist_label = dist_name
        if print_bic and len(dist_names)>1:
            dist_label += ' (BIC: %.0f)' % BIC
        ax.plot(x, pdf_fitted, linewidth=3, label=dist_label)

    if skew_fit:
        dist_name = 'skewnorm'
        skew = stats.skew(data)
        mu, std = data.mean(), data.std()
        loc, scale, shape = skewnormal_parms(mean=mu, stdev=std, skew=skew)
        #pdf = 2 * norm.pdf(x, loc=mu, scale=std) * norm.cdf(x * skew, loc=mu, scale=std)
        #pdf = skew_norm.pdf(x, skew, loc=mu, scale=std) # Create SkewNorm PDF
        pdf = pdf_skewnormal(x, location=loc, scale=scale, shape=shape)
        BIC = calculate_BIC(data, pdf, [shape,mu,std])
        dist_label = dist_name
        if print_bic and len(dist_names)>1:
            dist_label += ' (BIC: %.0f)' % BIC
        ax.plot(x, pdf, linewidth=3, label=dist_label)

    # Add all the labels, title, & legend!
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(prop={'size':12},loc=2)
    if xlim:
        ax.set_xlim(xlim)

    return ax

def plot_hist_groups(df, group_by, plot_col, plot=True,
                    logx=False, normalize=False, adjusted=False,
                    min_obs=20, tot_pop_col='ACSTOTPOP',
                    dist_names=[], skew_fit=False,
                    bins=20, cols=3, top_adj=0.9,
                    fig_title=None, xlabel=None, ylabel='Probability',
                    area_unit=None):
    """ Plots (inline) histograms and normal distribution fit for:
            log(variable) by a specific group/column
        Returns new dataframe by groups.

        Parameters:
            df (Pandas DataFrame) : Dataframe input
            group_by (str) : Column in df to group by
            plot_col (str) : Column to Plot Histograms of

        Optional Parameters:
            log (Boolean) : Take the log of data (Default = False)
            normialize (Boolean) : Subtract mean & divide stdev (Default = False)
            adjusted (Boolean) : Multiply by population proportion (Default = False)
            min_obs (int) : Minimum number of observations for each group to include (Default = 20)
            tot_pop_col (str) : Used in standardization (Default = ACSTOTPOP)
            dist_names ([list]) : Distributions to fit (Default = False)
            skew_fit (Boolean) : Whether to include Skew fit (Default = False)
            bins (int) : Number of bins for Histograms (Default = 20)
            cols (int) : Number of Columns for figure (Default = 3)
            fig_title (str) : Title of the whole figure (default none)
            xlabel, ylabel (str) : x and y labels of plots (Default: x = None, y='Probability')
            area_unit (str) : type of area unit for title
    """

    # Filter if we have less than min_obs
    df = df.groupby(group_by).filter(lambda x: len(x[plot_col]) > min_obs)
    # Group
    grouped = df.groupby(group_by)

    # Calulate dimensions and placement of figures
    tot = len(grouped.groups) + 1
    if not tot < cols:
        rows = tot//cols
    else:
        cols = tot
        rows = 1
    height = rows * 3


    if plot:
        # Create figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=(20,height))
        fig.tight_layout(pad=1.5, w_pad=2.5, h_pad=5.5)
        axes = axes.ravel()

    if fig_title and plot:
        # Add title and appropriate spacing
        fig.suptitle(fig_title, fontsize=24)
        plt.subplots_adjust(top=top_adj) # TODO: adjust this based on size of plot

    city_df = []
    for i, (name, group) in enumerate(grouped):
        #group = group[group[plot_col] > 0] #filter out 0's?
        data = group[plot_col].dropna()
        obs = len(data)

        if adjusted:
            mean = group[tot_pop_col].mean()
            data = data * (group[tot_pop_col]/mean)
            data = data.dropna()

        # Add all the labels, title, & legend!
        if not area_unit:
            area_unit = 'Obs.'
        if not xlabel:
            xlabel = plot_col
        title = name.split('-')[0] + ' (%s: %s)' % (area_unit, str(obs))

        # TODO: This is ugly, i should do calculation somewhere else rather than this if plot thing.
        if plot:
            axes[i] = plot_single_hist(data, ax=axes[i],
                            logx=logx, normalize=normalize,
                            dist_names=dist_names, skew_fit=skew_fit,
                            bins=bins,
                            title=title, xlabel=xlabel, ylabel=ylabel)


        if logx:
            data = np.log(data)
        city_mean = data.mean()
        city_var = data.var()
        city_std = data.std()
        city_df.append([name, city_mean, city_var, city_std])

    if logx:
        col_name = 'log(%s)'%plot_col
    if adjusted:
        col_mean_name = col_name + '_dist_adjmean'
        col_var_name = col_name + '_dist_adjvar'
        col_std_name = col_name + '_dist_adjstd'
    else:
        col_mean_name = col_name + '_dist_mean'
        col_var_name = col_name + '_dist_var'
        col_std_name = col_name + '_dist_std'
    return DataFrame(city_df, columns=['CITY_NAME', col_mean_name, col_var_name, col_std_name])


def plot_ols(df, x_col, y_col, logx=True, logy=False,
                    run_reg=True, print_reg=False,
                    title=None, xlabel=None, ylabel=None):
    """ Print out OLS regression and graph for a variable

        Returns regression results
    """
    df = df[df[x_col].notnull() & df[y_col].notnull()]

    if logx:
        x = np.log(df[x_col]).dropna()
    else:
        x = df[x_col]
    if logy:
        y = np.log(df[y_col]).dropna()
    else:
        y = df[y_col]

    if run_reg:
        results = ols_reg(x, y)
        intercept, slope = results.params
        line = intercept + slope * x

    # Create the plots
    plt.plot(x, y, 'o', label="Data")
    if run_reg:
        plt.plot(x, line, '-', lw = 2, label="OLS Reg")

    if not xlabel:
        if logx:
            xlabel = 'log(%s)' % x_col
        else:
            xlabel = x_col
    if not ylabel:
        if logy:
            ylabel = 'log(%s)'% y_col
        else:
            ylabel = y_col
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    if print_reg:
        print (results.summary())
        print ('\n' + 120 * '-' + 2 * '\n')

    return results
