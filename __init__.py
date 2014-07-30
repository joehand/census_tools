from __future__ import division

import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns

from dist_fits import create_bic_df
from income_bins import calc_inc_weights, plot_inc_bins
from plot_functions import plot_hist_groups
from skew_norm import skew_norm
#from us_census import *
from utils import group_by_city

# Set graph styles to white and large
sns.set_style("whitegrid")
sns.set_context("talk")
