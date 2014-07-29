from __future__ import division

import pandas as pd
from pandas import DataFrame, Series

from us_census import *
from plot_functions import *

# Set matplotlib variables
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
matplotlib.rcParams['font.size']=12
matplotlib.rcParams['savefig.dpi']=100
matplotlib.rcParams['figure.subplot.bottom']=.1
