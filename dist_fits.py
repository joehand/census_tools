
import matplotlib.pyplot as plt
import numpy as np
from numpy import log, sum
from pandas import DataFrame
import scipy.stats as stats
from scipy.stats import norm

from skew_norm import skew_norm

def calculate_BIC(data, pdf, param):
    NLL = -sum(log(pdf))
    n = len(data)
    k = len(param)
    BIC = k*log(n)+2*NLL # THIS ASSUMES LARGE N

    return BIC

def fit_dist(data, dist_name, return_BIC=True):
    dist = getattr(stats, dist_name)
    param = dist.fit(data)
    pdf = dist.pdf(data, *param[:-2], loc=param[-2], scale=param[-1])

    BIC = calculate_BIC(data, pdf, param)

    if return_BIC:
        return param, BIC
    else:
        return param


def _create_bic_df(df, group_by, analysis_col,
                    log=False, normalize=False, adjusted=False,
                    min_obs=20, tot_pop_col='ACSTOTPOP',
                    dist_names=[], skew_fit=False):

    # Filter if we have less than min_obs
    df = df.groupby(group_by).filter(lambda x: len(x[analysis_col]) > min_obs)
    # Group
    grouped = df.groupby(group_by)

    group_bics = []

    # TODO: Speed this up
    for i, (name, group) in enumerate(grouped):
        #group = group[group[analysis_col] > 0] #filter out 0's?
        data = group[analysis_col].dropna()
        obs = len(data)

        group_info = {'CITY_NAME':name}

        if adjusted:
            data = data * (group[tot_pop_col]/group[tot_pop_col].mean())
            data = data.dropna()

        if log:
            data = np.log(data)

        if normalize:
            data = (data - data.mean())/data.std()

        if len(dist_names) or skew_fit:
            xmin, xmax = data.min(), data.max()
            x = np.linspace(xmin, xmax, obs) # Create x vals

        for dist_name in dist_names:
            param, BIC = fit_dist(data, dist_name)
            group_info[analysis_col + '_' + dist_name + '_BIC'] = BIC

        if skew_fit:
            skew = stats.skew(data)
            mu, std = norm.fit(data)
            pdf = skew_norm.pdf(data, skew, loc=mu, scale=std)
            BIC = calculate_BIC(data, pdf, [skew, mu, std])
            group_info['skewnorm_BIC'] = BIC #TODO: Check this is correct

        group_bics.append(group_info)

    return DataFrame(group_bics) #.set_index('CITY_NAME', drop=False)

def plot_bic_ranks(df, group_by, analysis_col, percentage=True, **kwargs):
    COL_NAMES = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh']

    bic_df = _create_bic_df(df, group_by, analysis_col, **kwargs)
    rank = bic_df.rank(axis=1).filter(regex='_BIC$')

    bic_cols = rank.filter(regex='_BIC$').columns
    print bic_cols
    print rank.info()
    rank_counts = {col:rank[col].value_counts() for col in bic_cols}

    rank_counts = DataFrame(rank_counts).transpose().fillna(value=0)
    print rank_counts.columns
    rank_counts.columns = COL_NAMES[:len(rank_counts)]
    rank_counts = rank_counts.sort(columns=COL_NAMES[:len(rank_counts)],ascending=False)

    if percentage:
        rank_counts = (rank_counts/rank_counts.sum()) * 100

    ax = rank_counts.plot(kind='bar', title='Distribution BIC Ranks')
    ax.set_ylabel('Percentage')
    plt.show()
    ax = rank_counts['First'].plot(kind='bar', title='Distribution BIC First Place')
    ax.set_ylabel('Percentage')

    return rank_counts
