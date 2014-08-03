
from numpy import log
from pandas import DataFrame, Series
import scipy.stats as stats
from statsmodels import api as sm

def ols_reg(x, y, print_results=False):
    X = sm.add_constant(x, prepend=True) #Add a column of ones to allow the calculation of the intercept
    results = sm.OLS(y, X).fit()
    if print_results:
        print results.summary()
    return results

def _create_analysis_cols(df, analysis_cols, log_analysis):
    """

    """
    final_analysis_cols = []
    analysis_data = {}
    #TODO This could be a vectorized calculation rather than for loop
    for column in analysis_cols:
        temp_df = df   # TODO city_df[city_df[column] > 0] (should we do this?)
        data = temp_df[column].values

        if log_analysis:
            data = log(data)
            column = 'log(' + column + ')'

        # Add columns for mean, variance, and stdev
        analysis_data.update({
                column + '_mean' : data.mean(),
                column + '_variance': data.var(),
                column + '_stdev': data.std()
            })

        final_analysis_cols.extend([column + '_mean',
                column + '_variance',column + '_stdev'])

    return analysis_data, final_analysis_cols


def group_by_city(df, group_by='CITY_NAME', population_col='ACSTOTPOP',
                    sum_cols = [], analysis_cols=[],
                    min_obs=20, obs_name='BKGP', log_analysis=False, **kwargs):
    """ Returns dataframe with each row a city.

        Sums columns or returns mean, var, std
    """
    cities = []
    for name, city_df in df.groupby(group_by):
        # Check if # of obs (block groups) is > min_obs
        obs = len(city_df.index)
        if obs < min_obs:
            continue

        pop_col = city_df[population_col].values
        city_pop = pop_col.sum()

        # Create City Dict with name and total pop
        city = {
                group_by : name,
                population_col  : city_pop,
                obs_name + '_COUNT' : obs
            }

        # Add sum of each column in sum_cols to city dict
        city.update({column: city_df[column].values.sum() for column in sum_cols})

        analysis_data, final_analysis_cols = _create_analysis_cols(city_df, analysis_cols, log_analysis)
        city.update(analysis_data)
        cities.append(city)

    col_lists = [[group_by, population_col, obs_name + '_COUNT'],
                    sum_cols,
                    final_analysis_cols]
    df_cols = [item for sublist in col_lists for item in sublist]

    return DataFrame(cities, columns = df_cols)
