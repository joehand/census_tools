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
from plot_functions import plot_hist_groups
from skew_norm import skew_norm
#from us_census import *
from utils import group_by_city

# Set graph styles to white and large
sns.set_style("whitegrid")
sns.set_context("talk")
