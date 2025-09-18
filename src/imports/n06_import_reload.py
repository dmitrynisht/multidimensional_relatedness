from n06_import import *

import os

# Updating modules with the latest changes in case they were modified
from importlib import reload

import p01_raw_data_imports

reload(p01_raw_data_imports)
from p01_raw_data_imports import read_hs92_parquet_data, save_hs92_parquet_data
from p01_raw_data_imports import read_hs92_stata_data, save_hs92_stata_data
from p01_raw_data_imports import switch_to_dtype, custom_float_formatter, df_stats
from p01_raw_data_imports import restore_cache, staging_cache

import p06_mor_analysis

reload(p06_mor_analysis)
from p06_mor_analysis import merge_eci_to_gdp_by_year, merge_eci_to_gdp, merge_eci_n_gdp, filter_eci_to_gdp_n_order_year
from p06_mor_analysis import calculate_ECI_GDP_r_squared, calculate_yearly_eci_gdp_correlation, check_global_eci_convergence
from p06_mor_analysis import scatterplot_ECI_vs_GDP, lineplot_ECI_vs_GDP_correlation, plot_eci_convergence

print(f"os.getcwd():\n{os.getcwd()}", "=" * 60, sep="\n")
print("input_file_location:", input_file_location, "=" * 60, sep="\n")
print("output_file_location:", output_file_location, "=" * 60, sep="\n")

pass
