from __future__ import division

import re

import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns

from dist_fits import plot_bic_ranks
from income_bins import (calc_inc_weights, plot_inc_bins,
                    plot_incw_hist, plot_incw_beans, plot_incw_pop)
from plot_functions import plot_hist_groups, plot_single_hist,plot_ols
from skew_norm import skew_norm
from skew_normal import random_skewnormal, skewnormal_parms, pdf_skewnormal
#from us_census import *
from utils import group_by_city

#TODO: be able to set defaults like, population_col, city_col_name, etc.
#      this will need a class I think.
#      so make a class for app of this stuff. give it the api based on the imports above

# Set graph styles to white and large
sns.set_style("whitegrid")
sns.set_context("talk")
matplotlib.rc('text', usetex=True)
#plot syltes http://nbviewer.ipython.org/github/rasbt/matplotlib-gallery/blob/master/ipynb/publication.ipynb
