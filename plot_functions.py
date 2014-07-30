from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm

from skew_norm import skew_norm

def plot_hist_groups(df, group_by, plot_col,
                    log=False, normalize=False, standardize=False,
                    min_obs=20, tot_pop_col='ACSTOTPOP',
                    norm_fit=False, skew_fit=False,
                    bins=20, cols=3,
                    fig_title=None, xlabel=None, ylabel='Probability',
                    area_unit=None):
    """ Plots (inline) histograms and normal distribution fit for:
            log(variable) by a specific group/column

        Parameters:
            df (Pandas DataFrame) : Dataframe input
            group_by (str) : Column in df to group by
            plot_col (str) : Column to Plot Histograms of

        Optional Parameters:
            log (Boolean) : Take the log of data (Default = False)
            normialize (Boolean) : Subtract mean & divide stdev (Default = False)
            standardize (Boolean) : Multiply by population proportion (Default = False)
            min_obs (int) : Minimum number of observations for each group to include (Default = 20)
            tot_pop_col (str) : Used in standardization (Default = ACSTOTPOP)
            norm_fit (Boolean) : Whether to include Norm fit (Default = False)
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

    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20,height))
    fig.tight_layout(pad=1.5, w_pad=2.5, h_pad=5.5)
    axes = axes.ravel()

    if fig_title:
        # Add title and appropriate spacing
        fig.suptitle(fig_title, fontsize=24)
        plt.subplots_adjust(top=0.75)

    for i, (name, group) in enumerate(grouped):
        #group = group[group[plot_col] > 0] #filter out 0's?
        data = group[plot_col].dropna()
        obs = len(data)
        fit_label = 'Normal Fit'

        if standardize:
            # TODO: Make sure this is the right calculation
            # Should it be mean for group (City) or all data?
            data = data * (group[tot_pop_col]/group[tot_pop_col].mean())
            data = data.dropna()

        if log:
            data = np.log(data)
            fit_label = 'Log-Normal Fit'

        if normalize:
            data = (data - data.mean())/data.std()

        # Histogram of the data
        axes[i].hist(data.values, bins, normed=True, histtype="stepfilled", alpha=0.6)

        if norm_fit or skew_fit:
            mu, std = data.mean(), data.std()
            xmin, xmax = data.min(), data.max()
            x = np.linspace(xmin, xmax, obs) # Create x vals

        if norm_fit:
            # Plot the PDF.
            mu, std = norm.fit(data) # Fit a normal distribution to the data
            pdf = norm.pdf(x, loc=mu, scale=std) # Create Norm PDF
            axes[i].plot(x, pdf, linewidth=3, label=fit_label)

        if skew_fit:
            # Plot the PDF.
            fit_label = 'Skew ' + fit_label
            skew = stats.skew(data)
            #pdf = skew_norm.pdf(x, skew, loc=mu, scale=std) # Create SkewNorm PDF
            pdf = 2 * norm.pdf(x, loc=mu, scale=std) * norm.cdf(x * skew, loc=mu, scale=std)
            axes[i].plot(x, pdf, linewidth=3, label=fit_label)

        # Add all the labels, title, & legend!
        if not area_unit:
            area_unit = 'Obs.'
        if not xlabel:
            xlabel = plot_col
        title = name.split('-')[0] + ' (%s: %s)' % (area_unit, str(obs))
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(title)
        axes[i].legend(prop={'size':10},loc=2)
