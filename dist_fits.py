
import numpy as np
from numpy import log, sum
from pandas import DataFrame
import scipy.stats as stats
from scipy.stats import norm


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


def create_bic_df(df, group_by, analysis_col,
                    log=False, normalize=False, standardize=False,
                    min_obs=20, tot_pop_col='ACSTOTPOP',
                    dist_names=[], skew_fit=False):

    # Filter if we have less than min_obs
    df = df.groupby(group_by).filter(lambda x: len(x[analysis_col]) > min_obs)
    # Group
    grouped = df.groupby(group_by)

    group_bics = []

    for i, (name, group) in enumerate(grouped):
        #group = group[group[analysis_col] > 0] #filter out 0's?
        data = group[analysis_col].dropna()
        obs = len(data)

        group_info = {'CITY_NAME':name}

        if standardize:
            # TODO: Make sure this is the right calculation
            # Should it be mean for group (City) or all data?
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

        # TODO: Check this over
        if skew_fit:
            skew = stats.skew(data)
            mu, std = data.mean(), data.std()
            #pdf = skew_norm.pdf(x, skew, loc=mu, scale=std) # Create SkewNorm PDF
            pdf = 2 * norm.pdf(x, loc=mu, scale=std) * norm.cdf(x * skew, loc=mu, scale=std)
            BIC = calculate_BIC(data, pdf, [mu, std, skew])
            #group_info['skewnorm_BIC'] = BIC

        group_bics.append(group_info)

    return DataFrame(group_bics) #.set_index('CITY_NAME', drop=False)
