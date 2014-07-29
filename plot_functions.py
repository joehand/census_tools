import pandas as pd
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import pylab as P
import numpy as np

import statsmodels.api as sm

from math import pi, sqrt, copysign

import seaborn as sns

class Plot:

    colors = sns.color_palette()

    def _set_color(self, length):
        return sns.color_palette(sns.color_palette("hls", length))

    def _calc_params(self, arry):
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

        return x, mean, sigma

    def regression(self, df, x_col, y_col, log_x=True, log_y=False,
                        fig_title=None, xlabel=None, ylabel=None):
        """ Print out OLS regression and graph for a variable

            Returns regression results
        """

        df = df[df[x_col].notnull()]

        if log_x:
            x = np.log(df[x_col])
        else:
            x = df[x_col]
        if log_y:
            y = np.log(df[y_col])
        else:
            y = df[y_col]


        X = sm.add_constant(x, prepend=True) #Add a column of ones to allow the calculation of the intercept
        results = sm.OLS(y, X).fit()
        print results.summary()

        intercept, slope = results.params
        line = intercept + slope * x
        plt.plot(x, y, 'bo', color=self.colors[0], label="Data")
        plt.plot(x, line, 'r-', color=self.colors[2], lw = 2,
            label="OLS Reg")

        if not xlabel:
            if log_x:
                xlabel = 'log_' + x_col
            else:
                xlabel = x_col
        if not ylabel:
            if log_y:
                ylabel = 'log_' + y_col
            else:
                ylabel = y_col

        if fig_title:
            plt.title(fig_title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

        return results


    def plot_single(self, df, plot_col, bins=20, skew=True,
                        fig_title=None, xlabel=None, ylabel='Probability'):
        """ Plots single histogram and normal distribution fit for log(variable)

            Parameters:
                df (Pandas DataFrame) : Dataframe input
                plot_col (String) : Column to Plot Histograms of

            Optional Parameters:
                bins (int) : Number of bins for Histograms (Default = 20)
                skew (Boolean) : Whether to include Skew fit or not (Default = True)
                fig_title (string) : Title of the whole figure (default none)
                xlabel, ylabel (string) : x and y labels of plots (Default: x = None, y='Probability')
        """

        df = df[df[plot_col] > 0]
        num = df[plot_col].count()
        if num != 0:
            log_col = np.log(df[plot_col])
            x, mean, sigma = self._calc_params(log_col)

            plt.hist(log_col.values, bins, normed=True,  color=self.colors[0])
            plt.plot(x,mlab.normpdf(x,mean,sigma), label='Log-Normal', lw=3, color=self.colors[1])

            xpdf, y_min, skewness = self.calc_skew_pdf(log_col)

            if y_min is not None and skew:
                plt.plot(xpdf, y_min, label='Skew Log-Normal \n  (alpha = %s)'%str(skewness)[:6], lw=3,  color=self.colors[2])

            if not fig_title:
                fig_title = plot_col

            if not xlabel:
                xlabel = plot_col

            plt.legend(prop={'size':10},loc=2)
            plt.title(fig_title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

    def plot_groups(self, df, group_by, plot_col, log=False,
                        bins=20, min_obs=20, skew=True, cols=3,
                        fig_title=None, xlabel=None, ylabel='Probability', area_unit=None):
        """ Plots (inline) histograms and normal distribution fit for:
                log(variable) by a specific group/column

            Parameters:
                df (Pandas DataFrame) : Dataframe input
                group_by (String) : Column in df to group by
                plot_col (String) : Column to Plot Histograms of

            Optional Parameters:
                bins (int) : Number of bins for Histograms (Default = 20)
                min_obs (int) : Minimum number of observations for each group to include (Default = 20)
                skew (Boolean) : Whether to include Skew fit or not (Default = True)
                cols (int) : Number of Columns for figure (Default = 3)
                fig_title (string) : Title of the whole figure (default none)
                xlabel, ylabel (string) : x and y labels of plots (Default: x = None, y='Probability')
                area_unit (string) : type of area unit for title
        """

        grouped = df.groupby(group_by).filter(lambda x: len(x[plot_col]) > min_obs)
        grouped = grouped.groupby(group_by)

        # Calulate dimensions and placement of figures
        tot = len(grouped.groups) + 1
        if not tot < cols:
            rows = tot/cols
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
            plt.subplots_adjust(top=0.965)

        for i, (name, group) in enumerate(grouped):
            group = group[group[plot_col] > 0]
            num = group[plot_col].count()

            if num > min_obs:
                if log:
                    log_col = np.log(group[plot_col])
                else:
                    log_col = group[plot_col]
                x, mean, sigma = self._calc_params(log_col)
                density = stats.gaussian_kde(log_col)

                axes[i].hist(log_col.values, bins, normed=True, histtype="stepfilled", color=self.colors[0], alpha=0.7)
                axes[i].plot(x,mlab.normpdf(x,mean,sigma), linewidth=3, label='Log-Normal', color=self.colors[1])

                xpdf, y_min, skewness = self.calc_skew_pdf(log_col)
                if y_min is not None and skew:
                    axes[i].plot(xpdf, y_min, lw=3, label='Skew Log-Normal \n  (alpha = %s)'%str(skewness)[:6], color=self.colors[2])

                name = name.split('-')[0]

                if not area_unit:
                    area_unit = 'Obs.'
                title = name + ' (%s: %s)' % (area_unit, str(num))

                if not xlabel:
                    xlabel = plot_col

                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
                axes[i].set_title(title)
                axes[i].legend(prop={'size':10},loc=2)

            else:
                #print 'not enough obs for %s' % name
                pass

    def norm_tests(self, df, test_col, log=False):
        vals = df[test_col].values

        if log:
            vals = np.log(vals)

        # compute some statistics
        A2, sig, crit = stats.anderson(vals) # Anderson-Darling Test
        D, pD = stats.kstest(vals, "norm") # Kolmogorow-Smirnov Test
        W, pW = stats.shapiro(vals)  #Shapiro-Wilk Test


        print 70 * '-'
        print 'Normality Tests'
        print 70 * '-'
        print "  Kolmogorov-Smirnov test: D = %.2g  p = %.2f" % (D, pD)
        """
        # don't print this test for now.
        print "  Anderson-Darling test: A^2 = %.2f" % A2
        print "    significance  | critical value "
        print "    --------------|----------------"
        for j in range(len(sig)):
            print "    %.2f          | %.1f%%" % (sig[j], crit[j])
        """
        print "  Shapiro-Wilk test: W = %.2g p = %.2f" % (W, pW)
        print '\n'


    def plot_fits(self, df, plot_col, bins=20, skew=True, log=True,
                        dist_names = ['norm','genlogistic',
                            'loggamma'],
                        norm_test = False,
                        fig_title=None, xlabel=None, ylabel='Probability'):
        """ Plots single histogram and various distribution fit for log(variable)

            Parameters:
                df (Pandas DataFrame) : Dataframe input
                plot_col (String) : Column to Plot Histograms of

            Optional Parameters:
                bins (int) : Number of bins for Histograms (Default = 20)
                skew (Boolean) : Whether to include Skew fit or not (Default = True)
                fig_title (string) : Title of the whole figure (default none)
                xlabel, ylabel (string) : x and y labels of plots (Default: x = None, y='Probability')
        """
        if norm_test:
            self.norm_tests(df, plot_col, log=log)
            print 70 * '-'
            print 'Distribution Information & BIC Values'
            print 70 * '-'

        #df = df[df[plot_col] > 0]
        num = df[plot_col].count()
        if num != 0:
            if log is True:
                df = df[df[plot_col] > 0] #TODO: What do we want to do with 0's
                data_col = np.log(df[plot_col])
            else:
                data_col = df[plot_col]
            x, mean, sigma = self._calc_params(data_col)

            plt.hist(data_col.values, bins, normed=True, color='w')
            data = data_col.values

            for dist_name in dist_names:
                dist = getattr(stats, dist_name)
                param = dist.fit(data)
                pdf = dist.pdf(data, *param[:-2], loc=param[-2], scale=param[-1])

                NLL = -sum(np.log(pdf))
                k = len(param)
                n = len(data)
                BIC = k*np.log(n)+2*NLL # THIS ASSUMES LARGE N
                print dist_name
                print '\t Location: %.2g, Scale: %.2g' % (param[-2], param[-1])
                if param[:-2]:
                    print '\t Shape Params: %.2f' % param[:-2]
                print '\t NLL: %.0f' %(NLL)
                print '\t BIC: %.0f' %(BIC)
                print 70 * '_'

                pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
                label = dist_name + ' (BIC: %.0f)' % BIC
                plt.plot(x,pdf_fitted, label=label)
                #plt.xlim(data_col.min()*1.1,data_col.max()*1.1)

            if skew:
                xpdf, y_min, skewness = self.calc_skew_pdf(data_col)
                """
                #Skewed BIC value
                NLL = -sum(np.log(y_min))
                k = 3
                n = len(data)
                BIC = k*np.log(n)+2*NLL # THIS ASSUMES LARGE N
                print 'skew normal'
                print '\t Location: %.2g, Scale: %.2g' % (param[-2], param[-1])
                if param[:-2]:
                    print '\t Shape Params: %.2f' % param[:-2]
                print '\t NLL: %.0f' %(NLL)
                print '\t BIC: %.0f' %(BIC)
                print 70 * '_'
                """

                if y_min is not None:
                    plt.plot(xpdf, y_min, label='skew norm')

            if not fig_title:
                fig_title = plot_col

            if not xlabel:
                xlabel = plot_col

            plt.legend(prop={'size':10},loc=2)
            plt.title(fig_title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()

    def plots_collapsed(self, df, cols,
                            by_col = 'CITY', min_obs=100, bins=20, show_legend=True,
                            title=None, xlabel=None, ylabel='Probability'):

        """ Plots collapsed KDE for the a column of DF according to the another column

            Parameters:
                df (Pandas DataFrame) : Dataframe input
                cols (list) : Column(s) to plot on x axis
                by_col (String) : Column to group data by

            Optional Parameters:
                bins (int) : Number of bins for Histograms (Default = 20)
                min_obs (int) : Minimum number of observations for each group to include (Default = 20)
                title (string) : Title of the whole figure (default none)
                xlabel, ylabel (string) : x and y labels of plots (Default: x = None, y='Probability')
        """
        tot = len(cols)

        fig, axes = plt.subplots(nrows=tot*2, ncols=1)
        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        #fig = plt.figure()
        plt.subplots_adjust(hspace = 0.5, top=1.5)
        for i, col_name in enumerate(cols):
            df = df[df[col_name] > 0]
            num = df[col_name].count()

            length = len(pd.value_counts(df[by_col]).index) + 1 #add one for normal line
            colors = self._set_color(length)

            if num != 0:
                plt.subplot(tot, 1, i)

                for j, val in enumerate(pd.value_counts(df[by_col]).index):
                    # Get new DF based on our new group

                    df_new = df[df[by_col] == val]
                    log_col = np.log(df_new[col_name])

                    #make sure we have the minimum number of observations
                    if log_col.count() < min_obs:
                        continue


                    x, mean, sigma = self._calc_params(log_col)

                    #Calculate KDE (may need to adjust this for finer "bandwidth", see: http://milkbox.net/note/gaussian-kde-smoothed-histograms-with-matplotlib/
                    density = stats.gaussian_kde(log_col)

                    # Manually collapsing
                    collapsed = (log_col - mean)/sigma

                    xCollapsed, _, _ = self._calc_params(collapsed)

                    dCollapsed = stats.gaussian_kde(collapsed)

                    #Plot the Manual Collapsed
                    plt.plot(xCollapsed,dCollapsed(xCollapsed),label=val, alpha=0.5, lw=2, color=colors[j])

                x = np.linspace(-4,4,1000)

                plt.plot(x, mlab.normpdf(x,0,1), label='Normal', lw=3, color=colors[j+1])

                if not title:
                    title = col_name
                if not xlabel:
                    xlabel = col_name

                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)

                if show_legend:
                    plt.legend()

    def plots_collapsed_multiple(self, dfs, cols,
                            by_cols, min_obs=100, bins=20, show_legend=True,
                            title=None, xlabel=None, ylabel='Probability'):

        """ Plots collapsed KDE for the a column of DF according to the another column

            Parameters:
                dfs (list of Pandas DataFrame) : Dataframes input
                cols (list of String) : Column(s) to plot on x axis
                by_cols (list of String) : Column to group data by

            Optional Parameters:
                bins (int) : Number of bins for Histograms (Default = 20)
                min_obs (int) : Minimum number of observations for each group to include (Default = 20)
                title (string) : Title of the whole figure (default none)
                xlabel, ylabel (string) : x and y labels of plots (Default: x = None, y='Probability')
        """
        color_count = 0

        df_lengths = [len(pd.value_counts(df[by_cols[i]]).index) for i, df in enumerate(dfs)]
        length = sum(df_lengths) + 1 #add one for normal line
        colors = self._set_color(length)

        for i, df in enumerate(dfs):
            col_name = cols[i]
            by_col = by_cols[i]

            df = df[df[col_name] > 0]
            num = df[col_name].count()

            if num != 0:
                for j, val in enumerate(pd.value_counts(df[by_col]).index):
                    # Get new DF based on our new group

                    df_new = df[df[by_col] == val]
                    log_col = np.log(df_new[col_name])

                    #make sure we have the minimum number of observations
                    if log_col.count() < min_obs:
                        continue

                    x, mean, sigma = self._calc_params(log_col)

                    #Calculate KDE (may need to adjust this for finer "bandwidth", see: http://milkbox.net/note/gaussian-kde-smoothed-histograms-with-matplotlib/
                    density = stats.gaussian_kde(log_col)

                    # Manually collapsing
                    collapsed = (log_col - mean)/sigma

                    xCollapsed, _, _ = self._calc_params(collapsed)

                    dCollapsed = stats.gaussian_kde(collapsed)

                    #Plot the Manual Collapsed
                    plt.plot(xCollapsed,dCollapsed(xCollapsed),label=val, alpha=0.5, lw=2, color=colors[color_count])
                    color_count += 1

                x = np.linspace(-4,4,1000)

        plt.plot(x, mlab.normpdf(x,0,1), label='Normal', lw=3,  color=colors[color_count])

        if not title:
            title = col_name
        if not xlabel:
            xlabel = col_name

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show_legend:
            plt.legend()


    @staticmethod
    def _skewnormal_parms(mean=0.0, stdev=1.0, skew=0.0):
        if abs(skew) > Plot._skew_max():
            #print('Skewness must be between %.8f and %.8f' % (
            #                                        -Plot._skew_max(), Plot._skew_max()))
            return None, None, None
            """
                if we want to show all values regardless of bounds, use this
            skew = copysign(Plot._skew_max(), skew)
            """


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

    @staticmethod
    def _skew_max():
        beta = 2.0 - pi / 2.0
        #lim(delta, shape-> inf) = 1.0
        eps = sqrt(2.0 / pi)
        return beta * pow(eps, 3.0) / pow(1.0 - eps * eps, 3.0 / 2.0) - 1e-16

    @staticmethod
    def _pdf_skewnormal(x, location=0.0, scale=1.0, shape=0.0, normalize=True):
        if not normalize:
            return 2.0 * norm.pdf(x * norm.cdf(shape * x))
        if location and scale:
            t = (x - location) / scale
            return 2.0 / scale * norm.pdf(t) * norm.cdf(shape * t)
        else:
            return None

    def calc_skew_pdf(self, data):
        xpdf, mean, stdev = self._calc_params(data)
        num = data.count()
        skew = stats.skew(data)
        locm, scalem, shapem = self._skewnormal_parms(mean, stdev, skew)
        xpdf = np.linspace(mean - 4.0 * stdev, mean + 4.0 * stdev, num)
        y_min = self._pdf_skewnormal(xpdf, locm, scalem, shapem)
        #print 'Skew: %s, Mean %s, StDev: %s' % (str(skew), str(mean), str(stdev))
        return xpdf, y_min, skew
