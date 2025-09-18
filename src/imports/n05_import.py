import sys
import os
import gc  # garbage collector
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Path to 'src' folder is defined by 'config' and 'config_notebooks' files
from p01_raw_data_imports import read_hs92_parquet_data, save_hs92_parquet_data
from p01_raw_data_imports import read_hs92_stata_data, save_hs92_stata_data
from p01_raw_data_imports import switch_to_dtype, custom_float_formatter, df_stats
from p01_raw_data_imports import restore_cache, staging_cache

from p05_mor_validate import validate_eci_reflections_adjusted, validate_eci_reflections_big

# Assign the custom formatter to Pandas options
pd.options.display.float_format = custom_float_formatter

# Define the relative path to the 03_preprocessed folder
output_file_location = os.path.join("..", "data", "05_mor_silver")

# Define the relative path to the 02_ingested folder
input_file_location = os.path.join("..", "data", "04_mor_implemented")

print(f"os.getcwd():\n{os.getcwd()}", "=" * 60, sep="\n")
print("input_file_location:", input_file_location, "=" * 60, sep="\n")
print("output_file_location:", output_file_location, "=" * 60, sep="\n")

pass
