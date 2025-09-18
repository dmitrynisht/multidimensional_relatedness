"""
raw_data_imports.py

This module hold methods for data imports, used through the Masters project

Purpose:
Provide an ...

Author: Your Name
Date: YYYY-MM-DD
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle
from memory_profiler import profile

# from gp9k import report_variables


def staging_cache(*, f_quiet=False, **kwargs):
    """Pickling all variables passed as kwargs"""
    # Version Compatibility: Be aware that pickle files might not be compatible across different Python versions1.
    # Multiple Variables: If you need to store multiple variables, consider putting them in a dictionary or list before pickling

    # Storing multiple variables
    dump_file_location = kwargs.pop("output_file_location")
    dump_file_name = kwargs.pop("output_file_name", "staged_kwargs.pickle")
    dump_file_path = os.path.join(dump_file_location, dump_file_name)
    dumped_data = kwargs

    with open(dump_file_path, "wb") as file:
        pickle.dump(dumped_data, file)

    print(f"data staged:{[*dumped_data.keys()]}", "=" * 60, sep="\n")
    # try:
    #     with open(dump_file_path, "wb") as file:
    #         pickle.dump(dumped_data, file)

    #     print(f"data staged:{[*dumped_data.keys()]}", "=" * 60, sep="\n")
    # except FileNotFoundError:
    #     if f_quiet:
    #         # Create file at dump_file_path location
    #         os.makedirs(dump_file_location, exist_ok=True)
    #         with open(dump_file_path, "wb") as file:
    #             pickle.dump(dumped_data, file)
    #         print(f"data staged:{[*dumped_data.keys()]}", "=" * 60, sep="\n")
    #     else:
    #         raise FileNotFoundError(f"The file {dump_file_path} was not found!")

    pass


def restore_cache(**kwargs):
    """Restoring all variables from the pickle file"""

    # Restoring multiple variables
    dump_file_location = kwargs.pop("input_file_location", "")
    dump_file_name = kwargs.pop("input_file_name", "staged_kwargs.pickle")
    dump_file_path = os.path.join(dump_file_location, dump_file_name)

    try:
        with open(dump_file_path, "rb") as file:
            restored_data = pickle.load(file)
            print(f"data restored:{[*restored_data.keys()]}", "=" * 60, sep="\n")
    except FileNotFoundError:
        print(f"FileNotFoundError: The file {dump_file_path} was not found!")
        return
    except MemoryError:
        print("MemoryError: Not enough memory to load the file!")
        return

    return restored_data


def storing_dump(**kwargs):
    """Pickling all variables passed as kwargs
    Was planning to use in read_in_chunks() method"""
    # Version Compatibility: Be aware that pickle files might not be compatible across different Python versions1.
    # Multiple Variables: If you need to store multiple variables, consider putting them in a dictionary or list before pickling
    # Storing multiple variables
    file_location = kwargs["file_location"]
    chunk_iterator = kwargs["chunk_iterator"]
    file_sitc2 = kwargs["file_sitc2"]

    dump_file_name = "dumped_data.pickle"
    dump_file = "/".join([file_location, dump_file_name])
    dumped_data = {
        "chunk_iterator": chunk_iterator,
        "dump_file_location": file_location,
        "dump_file_name": dump_file_name,
        "dump_file": dump_file,
        "file_sitc2": file_sitc2,
    }
    with open(dump_file, "wb") as file:
        pickle.dump(dumped_data, file)

    # # Restoring multiple variables # actually didn't finish this part, wanted to have one function for saving cache into pickle file and another for restoring it
    # with open(dump_file, 'rb') as file:
    #     restored_data = pickle.load(file)

    # # checking if file exists before trying to read it
    # try:
    #     with open('your_file.txt', 'w') as file:
    #         content = file.read()
    #         # Process the file content here
    #         print("File content:", content)
    # except FileNotFoundError:
    #     print("The file doesn't exist. Skipping file reading.")
    #     # Continue with your code here, skipping the file reading step


def read_in_chunks():
    """Reading big files using pandas library
    utilizing pd.read_table() as context manager and setting chunksize argument"""

    # data_path = "data/01_raw/OEC Observatory of economic complexity/ReadMe-OEC.txt"
    # file_sitc2 = 'data/01_raw/OEC Observatory of economic complexity/trade_i_oec_a_sitc2.tsv'
    file_location = "data/01_raw/OEC Observatory of economic complexity"
    file_name = "trade_i_oec_a_sitc2.tsv"
    file_sitc2 = "/".join([file_location, file_name])

    # Specify the chunk size
    chunk_iterator = {
        "chunk_size": 100000,
        "chunk_count": 0,
        "total_rows": 0,
        "limit": 2,
        "year_start": 1962,
        "year_end": 1962,
        "chunk_year_min": 1962,
        "chunk_year_max": 1962,
        "years_count": 1,
        "rows_per_year_avg": 2000000,
    }
    with pd.read_table(
        file_sitc2, sep="\t", chunksize=chunk_iterator["chunk_size"]
    ) as reader:
        # Iterate over chunks using read_table
        for chunk in reader:
            # Process each chunk-dataframe here

            # print(chunk.head())

            chunk_iterator["chunk_count"] += 1
            chunk_iterator["total_rows"] += chunk.shape[0]

            # chunk_year_min, chunk_year_min valid only for debugging
            chunk_iterator["chunk_year_max"] = chunk["year"].max()
            chunk_iterator["chunk_year_min"] = chunk["year"].min()
            chunk_iterator["years_count"] += (
                1
                if chunk_iterator["chunk_year_max"] > chunk_iterator["chunk_year_min"]
                else 0
            )
            chunk_iterator["rows_per_year_avg"] = (
                chunk_iterator["total_rows"] / chunk_iterator["years_count"]
            )

            # if chunk_iterator['chunk_count'] > chunk_iterator['limit']:
            #     break

            pass

    print(chunk_iterator)
    pass


def read_with_PyArrow():
    """Reading big files using with PyArrow library
    installation:
        pip install pyarrow
    """

    pass


def switch_to_dtype(
    data=None,
    default_dtype="int16",
    dtype_dict=None,
    categorical_columns=None,
    f_convert_integers=True,
    int_columns=None,
):
    """Switching the data type of columns in the DataFrame
    when importing data from Stata file, categorical columns are imported as 'object' or 'int16' data types.
    This function switches the data type of the columns to 'category' data type.
    When saving the data to a Stata file, we want the 'category' data type converted back to 'int16' data type.

    Args:
        data (pd.DataFrame): The DataFrame to switch the data type of columns
        default_dtype (str): The default data type to switch the columns to if not specified in the dtype_dict
        dtype_dict (dict): A dictionary mapping columns to their desired data types, optional
        categorical_columns (list): The list of columns to switch to the 'category' data type, optional
        f_convert_integers (bool): Flag, indicates if convertion to integer of numeric columns is needed, optional

    Returns:
        data (pd.DataFrame): The DataFrame with the switched data type of columns
    """
    # Switching to the specified data type for categorical columns
    if not isinstance(data, pd.DataFrame):
        print("The input is not a DataFrame")
        return

    data_columns = data.columns

    if not default_dtype:
        print("The default data type is not specified")
        return

    if not dtype_dict:
        dtype_dict = {}
    elif not isinstance(dtype_dict, dict):
        print("The dtype_dict is not a dictionary")
        return

    if not categorical_columns:
        categorical_columns = ["country_id", "product_id", "year", "country_code"]

    if f_convert_integers:
        if not int_columns:
            if "export_value" in data_columns:
                int_columns = ["export_value"]
            else:
                f_convert_integers = False

    for column in categorical_columns:
        if column in data_columns:
            print(
                f"{column} column initial type:", data[column].dtype, "=" * 60, sep="\n"
            )
            to_dtype = dtype_dict.get(column, default_dtype)
            data[column] = data[column].astype(to_dtype)
            print(
                f"{column} column final type:", data[column].dtype, "=" * 60, sep="\n"
            )

    if f_convert_integers:
        for column in int_columns:
            data[column] = data[column].astype("int64")

    return data


# this decorator is used to estimate the memory usage of reading the file into dataframe
# @profile
def read_hs92_stata_data(
    file_location=None, file_name=None, columns=None, categorical_columns=None
):
    """Reading data from Stata file

    Args:
        file_location (str): The location of the file to read
        file_name (str): The name of the file to read
        columns (list): The list of columns to read from the file
        categorical_columns (list): The list of columns to switch to the 'category' data type, optional

    Returns:
        data (pd.DataFrame): The DataFrame read from the file
    """
    # Set default file location and name if not provided
    file_location = "" if not file_location else file_location
    file_name = (
        "hs92_country_country_product_year_4_2010_2014.dta"
        if not file_name
        else file_name
    )
    file_hs92 = os.path.join(file_location, file_name)

    try:
        data = pd.read_stata(file_hs92, convert_categoricals=False, columns=columns)
    except FileNotFoundError:
        print(f"FileNotFoundError: The file {file_hs92} was not found!")
        return
    except MemoryError:
        print("MemoryError: Not enough memory to load the file!")
        return

    # Switching to the 'category' data type for categorical columns
    data = switch_to_dtype(
        data, default_dtype="category", categorical_columns=categorical_columns
    )

    return data


def read_hs92_parquet_data(
    file_location=None,
    file_name=None,
    columns=None,
    categorical_columns=None,
    f_convert_dtype=True,
    **kwargs,
):
    """Reading data from Parquet file

    Args:
        file_location (str): The location of the file to read
        file_name (str): The name of the file to read
        columns (list): The list of columns to read from the file
        categorical_columns (list): The list of columns to switch to the 'category' data type, optional

    Returns:
        data (pd.DataFrame): The DataFrame read from the file
    """
    # Set default file location and name if not provided
    file_location = "" if not file_location else file_location
    file_name = "raw_hs92_exports_2010_2014.parquet" if not file_name else file_name
    file_path = os.path.join(file_location, file_name)

    try:
        # Read Parquet file with selected columns
        data = pd.read_parquet(file_path, columns=columns, engine="pyarrow")
    except FileNotFoundError:
        print(f"FileNotFoundError: The file {file_path} was not found!")
        return
    except MemoryError:
        print("MemoryError: Not enough memory to load the file!")
        return
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    # Convert categorical columns to 'category' dtype for efficiency
    if f_convert_dtype:
        default_dtype= kwargs["default_dtype"] if "default_dtype" in kwargs else "category"
        data = switch_to_dtype(
            data, default_dtype=default_dtype, categorical_columns=categorical_columns
        )

    return data


def save_hs92_stata_data(data, file_location=None, file_name=None, dtype_dict=None):
    """Saving data to a Stata file

    Args:
        data (pd.DataFrame): The DataFrame to save to a file
        file_location (str): The location to save the file to
        file_name (str): The name of the file to save the DataFrame to
        dtype_dict (dict): A dictionary mapping columns to their desired data types, optional
    """
    if not dtype_dict:
        dtype_dict = {"year": "int16"}
    elif not isinstance(dtype_dict, dict):
        print("The dtype_dict is not a dictionary")
        return
    else:
        dtype_dict["year"] = "int16"

    # Saving Stata files
    data_to_save = data.copy()

    # Switching to the 'category' data type for categorical columns
    data_to_save = switch_to_dtype(
        data_to_save, default_dtype="int16", dtype_dict=dtype_dict
    )

    # Save the DataFrame to a .dta file
    output_file_path = os.path.join(file_location, file_name)
    data_to_save.to_stata(output_file_path, write_index=False)


def save_hs92_parquet_data(
    data,
    file_location=None,
    file_name=None,
    dtype_dict=None,
    categorical_columns=None,
    compression="snappy",
    f_convert_dtype=True,
):
    """Saving data to a Parquet file

    Args:
        data (pd.DataFrame): The DataFrame to save to a file
        file_location (str): The location to save the file to
        file_name (str): The name of the file to save the DataFrame to
        dtype_dict (dict): A dictionary mapping columns to their desired data types, optional
        compression (str): Compression type for parquet ("snappy", "gzip", "brotli"), default is "snappy".
    """
    if not dtype_dict:
        dtype_dict = {"year": "int16"}
    elif not isinstance(dtype_dict, dict):
        print("The dtype_dict is not a dictionary")
        return
    else:
        dtype_dict["year"] = "int16"

    # Ensure file_name ends with .parquet
    if not file_name.endswith(".parquet"):
        file_name += ".parquet"

    # Making a copy of the dataset
    data_to_save = data.copy()

    # Convert categorical columns and optimize memory usage
    if f_convert_dtype:
        data_to_save = switch_to_dtype(
            data_to_save,
            default_dtype="int16",
            dtype_dict=dtype_dict,
            categorical_columns=categorical_columns,
        )

    # Save the DataFrame to a Parquet file
    output_file_path = os.path.join(file_location, file_name)
    data_to_save.to_parquet(
        output_file_path, engine="pyarrow", index=False, compression=compression
    )

    # print(f"Data successfully saved to {output_file_path} using {compression} compression.")
    pass


def custom_float_formatter(x):
    """Custom function to format float values based on their size"""

    if abs(x) <= 1 and abs(x) > 0:  # Small numbers in the range [-1, +1]
        # Standard fixed-point notation with 6 decimal places
        return f"{x:.6f}"

    # elif abs(x) >= 1e6:  # Large numbers (≥ 1,000,000 or ≤ -1,000,000)
    #     # Fixed-point notation with commas and 2 decimals
    #     return f"{x:,.2f}"

    elif abs(x) > 999:  # Large numbers (≥ 1,000,000 or ≤ -1,000,000)
        # Fixed-point notation with commas and 2 decimals
        return f"{x:,.2f}"

    # elif abs(x) > 1:  # Large numbers (≥ 1,000,000 or ≤ -1,000,000)
    #     # Fixed-point notation with commas and 2 decimals
    #     return f"{x:.2f}"

    else:  # Other numbers
        # Standard fixed-point notation with 2 decimals
        return f"{x:.2f}"


def df_stats(df=None, df_name=None):
    """Calculating basic statistics for the dataframe"""

    print(["df" if not df_name else df_name, type(df)], "=" * 60, sep="\n")
    if df is None:
        return

    if not isinstance(df, pd.DataFrame):
        print("The input is not a DataFrame")
        return

    print("shape:", df.shape, "=" * 60, sep="\n")
    print("data types:", df.dtypes, "=" * 60, sep="\n")
    print("columns:", [*df.columns], "=" * 60, sep="\n")
    print(df.describe(), "=" * 60, sep="\n")

    memory_usage = df.memory_usage(deep=True)
    try:
        total_memory_usage = memory_usage.sum()
        print(f"Total memory usage: {total_memory_usage} bytes", "=" * 60, sep="\n")
    except:
        print(f"memory_usage: {memory_usage}", "=" * 60, sep="\n")

    # # Create a DataFrame
    # data = {
    #     'A': [1, 2, 3, 4, 5],
    #     'B': [10, 20, 30, 40, 50],
    #     'C': [100, 200, 300, 400, 500]
    # }
    # df = pd.DataFrame(data)

    # # Calculate basic statistics
    # stats = {
    #     'mean': df.mean(),
    #     'median': df.median(),
    #     'std': df.std(),
    #     'min': df.min(),
    #     'max': df.max()
    # }

    # print(stats)

    pass


def test_staging_cache():
    """Testing the staging_cache() function"""
    print("Testing the staging_cache() function", "=" * 60, sep="\n")

    # Define the relative path to the 03_preprocessed folder
    output_file_location = os.path.join("data", "03_preprocessed")
    dummy_var = 42
    input_file_location = os.path.join("data", "03_preprocessed")
    # staging_cache(output_file_location=output_file_location, dummy_var=dummy_var)
    restore_cache(input_file_location=input_file_location)

    pass


def test_staging_paruet():
    print("Testing the test_staging_paruet() function", "=" * 60, sep="\n")

    # Define the relative path to the 02_ingested folder
    output_file_location = os.path.join("data", "02_ingested")

    # Define the relative path to the 01_raw folder
    input_file_location = os.path.join("data", "01_raw")

    input_file_name = "product_hs92.csv"
    # , usecols=['product_id', 'code', 'name_short_en', 'product_level', 'top_parent_id', 'product_id_hierarchy'])
    # reading the csv file, renaming 'name_short_en' to 'product_name'
    product_df = pd.read_csv(os.path.join(input_file_location, input_file_name))
    product_df.rename(
        columns={"name_short_en": "product_name", "code": "product_code"}, inplace=True
    )

    dtype_dict = {
        "product_id": "int16",
        "top_parent_id": "int16",
        "product_level": "category",
    }
    categorical_columns = ["product_id", "top_parent_id", "product_level"]

    # Save the raw exports Data to a .parquet file
    output_file_name = "product_hierarchy.parquet"
    save_hs92_parquet_data(
        data=product_df,
        file_location=output_file_location,
        file_name=output_file_name,
        categorical_columns=categorical_columns,
        dtype_dict=dtype_dict,
    )

    pass


def main(argv=None):
    if argv is None:
        argv = sys.argv

    s = "start"

    # test comes here

    # test_staging_cache()

    test_staging_paruet()

    # read_hs92_stata_data()

    s = "stop"


if __name__ == "__main__":
    sys.exit(main())
