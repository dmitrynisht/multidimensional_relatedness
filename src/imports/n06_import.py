import sys
import os
import gc  # garbage collector
import pandas as pd

# Path to 'src' folder is defined by 'config' and 'config_notebooks' files
from p01_raw_data_imports import read_hs92_parquet_data, save_hs92_parquet_data
from p01_raw_data_imports import read_hs92_stata_data, save_hs92_stata_data
from p01_raw_data_imports import switch_to_dtype, custom_float_formatter, df_stats
from p01_raw_data_imports import restore_cache, staging_cache

from p06_mor_analysis import merge_eci_to_gdp_by_year, merge_eci_to_gdp, merge_eci_n_gdp, filter_eci_to_gdp_n_order_year
from p06_mor_analysis import calculate_ECI_GDP_r_squared, calculate_yearly_eci_gdp_correlation, check_global_eci_convergence
from p06_mor_analysis import scatterplot_ECI_vs_GDP, lineplot_ECI_vs_GDP_correlation, plot_eci_convergence

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# for calculating the correlation coefficient R-squared
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assign the custom formatter to Pandas options
pd.options.display.float_format = custom_float_formatter

# Define the relative path to the output data folder
output_file_location = os.path.join("..", "data", "06_mor_gold")

# Define the relative path to the input data folder
input_file_location = os.path.join("..", "data", "05_mor_silver")

print(f"os.getcwd():\n{os.getcwd()}", "=" * 60, sep="\n")
print("input_file_location:", input_file_location, "=" * 60, sep="\n")
print("output_file_location:", output_file_location, "=" * 60, sep="\n")

pass
