
from numpy import log
from pandas import DataFrame

def group_by_city(df, group_by='CITY_NAME', population_col='ACSTOTPOP',
                    sum_cols = ['ACSHINC_TOTAL'], analysis_cols=[],
                    min_obs=20, obs_name='BKGP', log_analysis=False):
    """ Returns dataframe with each row a city.
        Sums columns or returns mean, var, std

    """

    cities = []
    for name, city_df in df.groupby(group_by):
        # Check if # of obs (block groups) is > min_obs
        obs = len(city_df.index)
        if obs < min_obs:
            continue

        city_pop = city_df[population_col].sum()

        # Create City Dict with name and total pop
        city = {
                group_by : name,
                population_col  : city_pop,
                obs_name + '_COUNT' : obs
            }

        # Add sum of each column in sum_cols to city dict
        city.update({column: city_df[column].sum() for column in sum_cols})

        #TODO This could be a vectorized calculation rather than for loop
        for column in analysis_cols:
            temp_df = city_df   # TODO city_df[city_df[column] > 0] (should we do this?)
            data = temp_df[column]

            if log_analysis:
                data = log(data)
                column = 'log(' + column + ')'

            # Add columns for mean, variance, and stdev
            city.update({
                    column + '_mean' : data.mean(),
                    column + '_variance': data.var(),
                    column + '_stdev': data.std()
                })

            final_analysis_cols = [column + '_mean',
                    column + '_variance',column + '_stdev']

            """
            TODO: check this.
            ? - do i multiple by weight first, then get mean or get mean then multiply weight?

            weight = (temp_df[population_col] / city_pop)
            weighted_col = (temp_df[column] * weight)
            _, mean, sigma, variance = calc_params(weighted_col)

            mean = np.log(weighted_col.sum()) #sum the column and get the mean
            #sigma = np.log(sigma)
            #variance = np.log(variance)

            city.update({
                    'log_' + column + '_norm_mean' : mean,
                    'log_' + column + '_norm_variance': variance,
                    'log_' + column + '_norm_sigma': sigma,
                    'sum_weight' : weight,
                })
            """

        cities.append(city)

    col_lists = [[group_by, population_col, obs_name + '_COUNT'],
                    sum_cols,
                    final_analysis_cols]
    df_cols = [item for sublist in col_lists for item in sublist]

    return DataFrame(cities, columns = df_cols)
