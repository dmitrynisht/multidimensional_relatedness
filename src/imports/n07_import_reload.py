from n07_import import *

import os

# Updating modules with the latest changes in case they were modified
from importlib import reload
import p01_raw_data_imports

reload(p01_raw_data_imports)
from p01_raw_data_imports import read_hs92_parquet_data, save_hs92_parquet_data
from p01_raw_data_imports import read_hs92_stata_data, save_hs92_stata_data
from p01_raw_data_imports import switch_to_dtype, custom_float_formatter, df_stats
from p01_raw_data_imports import restore_cache, staging_cache

# Assign the custom formatter to Pandas options
pd.options.display.float_format = custom_float_formatter

print(f"os.getcwd():\n{os.getcwd()}", "=" * 60, sep="\n")
print("input_file_location:", input_file_location, "=" * 60, sep="\n")
print("output_file_location:", output_file_location, "=" * 60, sep="\n")

pass
