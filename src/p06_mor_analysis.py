"""
p06_mor_analysis.py
(Method of Reflections, analysis)

This module is a collection of methods used mainly in n06_MOR_visual to analyse the implementation of ECI,
using Method of Reflections(MOR). And some testing methods, which are not expected to be imported elsewhere.

Purpose:
Provide an ...

Author: Dzmitry Nisht
Date: YYYY-MM-DD
"""

import sys
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# from gp9k import report_variables

from p01_raw_data_imports import read_hs92_stata_data, custom_float_formatter, df_stats

pass


def merge_eci_n_gdp(ECI, GDP, n):
    """
    Merging the ECI and GDP dataframes for the given order of reflection n
    Args:
        ECI (DataFrame): The ECI dataframe
        GDP (DataFrame): The GDP dataframe
        n (int): The order of reflection
    Returns:
        ECI_GDP (DataFrame): The merged ECI and GDP dataframe
    """
    # Subsetting the ECI dataframe to only include the ECI_n index
    ECI_n = ECI[["country_code", "year", f"ECI_{n}"]].copy()

    # Renaming the ECI_n index to 'ECI'
    ECI_n.rename(columns={f"ECI_{n}": "ECI"}, inplace=True)

    # Merging the ECI_n index with the GDP dataframe
    ECI_GDP = GDP.merge(ECI_n, on=["country_code", "year"], how="left")

    return ECI_GDP


def merge_eci_to_gdp_by_year_v01(ECI, GDP, n, year):
    """
    Merging the ECI and GDP dataframes for the given order of reflection n and year
    Args:
        ECI (DataFrame): The ECI dataframe
        GDP (DataFrame): The GDP dataframe
        n (int): The order of reflection
    Returns:
        ECI_GDP (DataFrame): The merged ECI and GDP dataframe for given year and order of reflection
    """
    if not isinstance(ECI, pd.DataFrame):
        print("The ECI input is not a DataFrame")
        return

    if not isinstance(GDP, pd.DataFrame):
        print("The GDP input is not a DataFrame")
        return

    # Getting the non-numeric columns of the ECI dataframe
    categorical_columns = ECI.select_dtypes(exclude=["number"]).columns

    # Subsetting the ECI dataframe to only include the ECI_n index
    ECI_n = ECI[[*categorical_columns, f"ECI_{n}"]].copy()

    # Renaming the ECI_n index to 'ECI'
    ECI_n.rename(columns={f"ECI_{n}": "ECI"}, inplace=True)

    # Merging the ECI_n index with the GDP dataframe
    # ECI_GDP = ECI_n.merge(GDP, on=['country_code', 'year'], how='left')
    ECI_GDP = ECI_n.merge(GDP, on=["country_id", "year"], how="left")
    ECI_GDP = ECI_GDP[ECI_GDP["year"] == str(year)]

    return ECI_GDP


def merge_eci_to_gdp_by_year(ECI, GDP, n, year):
    """
    Merging the ECI and GDP dataframes for the given order of reflection n and year
    Args:
        ECI (DataFrame): The ECI dataframe
        GDP (DataFrame): The GDP dataframe
        n (int): The order of reflection
    Returns:
        ECI_GDP (DataFrame): The merged ECI and GDP dataframe for given year and order of reflection
    """
    if not isinstance(ECI, pd.DataFrame):
        print("The ECI input is not a DataFrame")
        return

    if not isinstance(GDP, pd.DataFrame):
        print("The GDP input is not a DataFrame")
        return

    # Merging the ECI_n index with the GDP dataframe
    ECI_GDP = merge_eci_to_gdp(ECI, GDP, n)
    ECI_GDP_n_year = ECI_GDP[ECI_GDP["year"] == year]

    return ECI_GDP_n_year


def merge_eci_to_gdp(ECI, GDP):
    """
    Merging the ECI and GDP dataframes by country_id and year
    Args:
        ECI (DataFrame): The ECI dataframe
        GDP (DataFrame): The GDP dataframe
    Returns:
        ECI_GDP (DataFrame): The merged ECI and GDP dataframe for all years and orders of reflection
    """
    if not isinstance(ECI, pd.DataFrame):
        print("The ECI input is not a DataFrame")
        return

    if not isinstance(GDP, pd.DataFrame):
        print("The GDP input is not a DataFrame")
        return

    # Getting the numeric columns of the ECI dataframe
    numeric_columns = ECI.select_dtypes(include=["number"]).columns

    # Exclude the "country_id" column by name
    numeric_columns = numeric_columns.drop(["country_id", "year"])

    # Select categorical columns by excluding the numeric columns
    categorical_columns = ECI.columns.difference(numeric_columns).to_list()

    # Merging the ECI_n index with the GDP dataframe
    ECI_GDP = ECI.merge(GDP, on=categorical_columns, how="left")

    return ECI_GDP


def eci_to_gdp_n_order(ECI_GDP, n):
    """
    Merging the ECI and GDP dataframes for the given order of reflection n
    Args:
        ECI (DataFrame): The ECI dataframe
        GDP (DataFrame): The GDP dataframe
        n (int): The order of reflection
    Returns:
        ECI_GDP (DataFrame): The merged ECI and GDP
    """
    if not isinstance(ECI_GDP, pd.DataFrame):
        print("The ECI input is not a DataFrame")
        return

    # Getting the numeric columns of the ECI dataframe
    numeric_columns = ECI_GDP.select_dtypes(include=["number"]).columns

    # Exclude the "country_id" column by name
    numeric_columns = numeric_columns.drop(["country_id", "year"])

    # Select categorical columns by excluding the numeric columns
    categorical_columns = ECI_GDP.columns.difference(numeric_columns).to_list()

    # Subsetting the ECI dataframe to only include the ECI_n index
    ECI_GDP_n = ECI_GDP[[*categorical_columns, f"ECI_{n}"]].copy()

    # # Renaming the ECI_n index to 'ECI'
    # ECI_n.rename(columns={f"ECI_{n}": "ECI"}, inplace=True)

    return ECI_GDP_n


def filter_eci_to_gdp_n_order_year(ECI_GDP, year, n):
    """
    Filtering the ECI and GDP dataframes for the given order of reflection n and year
    Args:
        ECI_GDP (DataFrame): The merged ECI_GDP dataframe
        year (int): The year to filter the data
        n (int): The order of reflection
    Returns:
        ECI_n (DataFrame): The filtered ECI_GDP
    """
    if not isinstance(ECI_GDP, pd.DataFrame):
        print("The ECI input is not a DataFrame")
        return

    # Getting the numeric columns of the ECI dataframe
    numeric_columns = ECI_GDP.select_dtypes(include=["number"]).columns

    # # Exclude the "country_id" column by name
    # numeric_columns = numeric_columns.drop(["country_id", "year"])

    # Select categorical columns by excluding the numeric columns
    categorical_columns = ECI_GDP.columns.difference(numeric_columns).to_list()

    # Subsetting the ECI dataframe to only include the ECI_n index for specified year
    ECI_GDP_n = ECI_GDP[ECI_GDP["year"] == year]
    ECI_GDP_n = ECI_GDP_n[
        [*categorical_columns, f"ECI_{n}", "log_gdp_per_capita"]
    ].reset_index(drop=True)

    return ECI_GDP_n


def scatterplot_ECI_vs_GDP(
    ECI_GDP: pd.DataFrame, 
    year=None, 
    n=12, 
    f_use_labels=False
):
    """
    Plots ECI_n (order of reflection) vs log_gdp_per_capita across multiple years as subplots.

    Parameters:
    - ECI_GDP: DataFrame containing columns ['year', 'ECI_n', 'log_gdp_per_capita']
    - year: None (default) to visualize all years, or a list of specific years.
    - n: Order of reflection to select a specific "ECI_{n}" column.

    The function creates a subplot for each selected year.
    """
    # Select the required column dynamically
    eci_column = f"ECI_{n}"

    if eci_column not in ECI_GDP.columns:
        raise ValueError(f"Column '{eci_column}' not found in DataFrame.")

    # Select years based on user input or take all available years
    unique_years = sorted(ECI_GDP["year"].unique()) if year is None else sorted(year)

    # Define number of rows and columns for subplots
    num_years = len(unique_years)
    cols = min(3, num_years)  # Max 3 columns
    rows = (num_years // cols) + (num_years % cols > 0)  # Compute required rows

    if num_years == 0:
        raise ValueError("No matching years found in DataFrame.")

    figsize = (10, 6) if num_years == 1 else (cols * 5, rows * 4)

    # Define subplot grid
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=figsize, constrained_layout=True
    )
    axes = axes.flatten() if num_years > 1 else [axes]
    fig.set_tight_layout(True)

    # Create scatter plots with regression lines
    for idx, year in enumerate(unique_years):
        df_year = ECI_GDP[ECI_GDP["year"] == year]
        ax = axes[idx]

        sns.regplot(
            data=df_year,
            x=eci_column,
            y="log_gdp_per_capita",
            ax=ax,
            ci=None,
            scatter_kws={
                "s": 30,
                "color": "blue",
                "alpha": 0.7,
                "label": "Data Points",
            },  # Customize scatter points
            line_kws={
                "color": "red",
                "label": "Regression Line",
            },  # Customize the regression line
        )

        # Add labels using apply
        if f_use_labels:
            print("initial label font size:", mpl.rcParams["font.size"])
            df_year.apply(
                lambda row: ax.text(
                    row[eci_column],
                    row["log_gdp_per_capita"],
                    row["country_code"],
                    fontsize=6,
                    ha="center",
                    va="bottom",
                ),
                axis=1,
            )
            print("applied label font size:", 6)

        # Formatting
        ax.set_title(f"ECI_{n} vs Log GDP per Capita - {year}")
        ax.set_xlabel(f"ECI_{n}")
        ax.set_ylabel("Log GDP per Capita")
        ax.grid(True)

    # Hide any extra empty subplots
    for ax in axes[num_years:]:
        ax.set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_eci_vs_gdp_v2(
    ECI_GDP, 
    year=None, 
    n=12, 
    f_use_labels=False
):
    """
    Visualizes ECI_n (n-th order of reflection) vs log_gdp_per_capita across multiple years using seaborn subplots.

    Parameters:
    - ECI_GDP (pd.DataFrame): The dataframe containing ECI and GDP data.
    - year (list or None): List of specific years to plot, or None to plot all available years.
    - n (int): Order of reflection to select "ECI_n" column for visualization.

    Returns:
    - None: Displays the subplots.
    """
    # Select the required column dynamically
    eci_column = f"ECI_{n}"

    # Select years based on user input or take all available years
    unique_years = sorted(ECI_GDP["year"].unique()) if year is None else sorted(year)

    # Define number of rows and columns for subplots
    num_years = len(unique_years)
    cols = min(3, num_years)  # Max 3 columns
    rows = (num_years // cols) + (num_years % cols > 0)  # Compute required rows
    print(f"cols={cols}, rows={rows}")

    # Set up the figure and axis
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True
    )
    axes = axes.flatten() if num_years > 1 else [axes]  # Flatten if single row

    # Loop through each selected year and plot
    for idx, year in enumerate(unique_years):
        df_year = ECI_GDP[ECI_GDP["year"] == year]
        ax = axes[idx]
        r_squared = calculate_ECI_GDP_r_squared(df_year, n)

        sns.regplot(
            data=df_year,
            x=eci_column,
            y="log_gdp_per_capita",
            ax=ax,
            ci=None,
            scatter_kws={
                "s": 30,
                "color": "blue",
                "alpha": 0.7,
                # "label": "Data Points",
            },  # Customize scatter points
            line_kws={
                "color": "red",
                "label": f"R² = {r_squared:.3f}",
            },  # Customize the regression line
        )

        # Add labels using apply (after regplot)
        if f_use_labels:
            df_year.apply(
                lambda row: ax.annotate(
                    row["country_code"],
                    (row[eci_column], row["log_gdp_per_capita"]),
                    xytext=(1, 3),  # offset in points
                    textcoords='offset points',
                    fontsize=8,
                    ha='left',
                    va='bottom'
                ),
                axis=1,
            )

        # Formatting
        # ax.set_title(f"ECI_{n} vs Log GDP per Capita - {year}")
        ax.legend()
        ax.set_xlabel(f"ECI_{n}")
        ax.set_ylabel("Log GDP per Capita")
        ax.grid(False)

    # Hide any extra empty subplots
    for ax in axes[num_years:]:
        ax.set_visible(False)

    # Show the final plot
    plt.show()


def plot_eci_convergence(ECI_GDP, country_codes=None, skip_eci_orders=None):
    """
    Plots ECI order of reflection convergence for given countries.

    Parameters:
    - ECI_GDP: DataFrame containing ECI calculations.
    - country_ids: List of country IDs to visualize. If None, defaults to ["USA"].

    Returns:
    - Matplotlib plot of ECI values over increasing orders of reflection.
    """
    # Default country list if none provided
    if country_codes is None:
        country_codes = ["USA"]

    if skip_eci_orders is None:
        skip_eci_orders = []
    else:
        skip_eci_orders = ["ECI_" + str(order) for order in skip_eci_orders]

    valid_codes = set(ECI_GDP["country_code"].unique())
    country_codes = set(country_codes)

    # Extract country_codes which are not valid_codes, using set intersection
    invalid_codes = country_codes.difference(valid_codes)

    # Extract valit country_codes
    country_codes = country_codes.intersection(valid_codes)

    for invalid_code in invalid_codes:
        print(f"Warning: invalid country_code {country_code}. Skipping...")
        continue

    # Extract ECI columns for even orders
    eci_columns = [col for col in ECI_GDP.columns if col.startswith("ECI_")]
    even_eci_columns = sorted(
        [
            col
            for col in eci_columns
            if int(col.split("_")[1]) % 2 == 0 and col not in skip_eci_orders
        ],
        key=lambda x: int(x.split("_")[1]),  # Sort by numeric value
    )
    print(even_eci_columns)
    orders = [int(col.split("_")[1]) for col in even_eci_columns]

    # Set up the plot
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")

    # Loop through each country and plot its ECI values
    for country_code in country_codes:
        df_country = ECI_GDP[ECI_GDP["country_code"] == country_code]

        eci_values = df_country[even_eci_columns].values.flatten()
        sns.lineplot(
            x=orders, y=eci_values, marker="o", linestyle="-", label=country_code
        )

    # Formatting
    plt.xlabel("Order of Reflection (Even)")
    plt.ylabel("ECI Value")
    plt.title("ECI Convergence Across Orders of Reflection")
    plt.legend(title="Country ID")
    plt.grid(True)

    # Show the plot
    plt.show()


def check_global_eci_convergence(ECI_GDP, threshold=0.1):
    """
    Checks the convergence of ECI across all countries by analyzing global mean and standard deviation.

    Parameters:
    - ECI_GDP: DataFrame containing ECI calculations.
    - threshold: Convergence threshold (default: 0.01 for 1%).

    Returns:
    - DataFrame summarizing global ECI convergence.
    - Plot of ECI convergence trend.
    """
    # Extract even ECI columns, skipping ECI_0
    eci_columns = [col for col in ECI_GDP.columns if col.startswith("ECI_")]
    even_eci_columns = sorted(
        [
            col
            for col in eci_columns
            if int(col.split("_")[1]) % 2 == 0 and col != "ECI_0"
        ],
        key=lambda x: int(x.split("_")[1]),  # Sort by numeric value
    )
    orders = [int(col.split("_")[1]) for col in even_eci_columns]
    print(even_eci_columns)
    # Compute global statistics across countries
    global_mean_eci = ECI_GDP[even_eci_columns].mean()
    global_std_eci = ECI_GDP[even_eci_columns].std()

    # Compute relative change in mean ECI (ΔECI_n)
    delta_eci = np.abs(np.diff(global_mean_eci)) / np.abs(global_mean_eci[:-1])

    # Check if the ECI has converged (ΔECI_n < threshold)
    converged = np.max(delta_eci) < threshold or global_std_eci.iloc[-1] < threshold

    # Store results in a DataFrame
    convergence_df = pd.DataFrame(
        {
            "Order of Reflection": orders,
            "Mean ECI": global_mean_eci.values,
            "Std Dev ECI": global_std_eci.values,
            "ΔECI_n": np.append([np.nan], delta_eci),  # First order has no ΔECI_n
        }
    )

    # Plot ECI Mean and Std Dev across orders
    fig, ax1 = plt.subplots(figsize=(8, 5))

    sns.lineplot(
        x=orders,
        y=global_mean_eci,
        marker="o",
        linestyle="-",
        ax=ax1,
        label="Mean ECI",
        color="blue",
    )
    ax1.set_xlabel("Order of Reflection (Even, Excluding ECI_0)")
    ax1.set_ylabel("Mean ECI", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Add secondary y-axis for standard deviation
    ax2 = ax1.twinx()
    sns.lineplot(
        x=orders,
        y=global_std_eci,
        marker="s",
        linestyle="--",
        ax=ax2,
        label="Std Dev ECI",
        color="red",
    )
    ax2.set_ylabel("Standard Deviation of ECI", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Add a horizontal threshold line
    ax2.axhline(
        threshold,
        color="gray",
        linestyle="dashed",
        linewidth=1,
        label="Convergence Threshold",
    )

    # Titles and Legends
    ax1.set_title("Global ECI Convergence Across Orders of Reflection")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.grid(True)
    plt.show()

    return convergence_df, converged


def calculate_ECI_GDP_r_squared(df_eciorder_year, n):
    """
    # Calculating the coefficient of determination, R-squared (R²), to assess the strength of the relationship between the "GDP per capita" and "ECI"
    # represented in scatterplot above
    """
    # Select the required column
    eci_column = f"ECI_{n}"

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(df_eciorder_year[[eci_column]], df_eciorder_year["log_gdp_per_capita"])

    # Calculate the R-squared value
    return model.score(
        df_eciorder_year[[eci_column]], df_eciorder_year["log_gdp_per_capita"]
    )


def calculate_yearly_eci_gdp_correlation(ECI_GDP, n_order):
    """
    Efficiently computes yearly correlation between multiple ECI_n columns and log_gdp_per_capita.

    Parameters:
        ECI_GDP (pd.DataFrame): DataFrame containing 'year', 'country_id', 'ECI_n' columns, and 'log_gdp_per_capita'.
        n_order (range): Range of ECI orders to compute correlation for (e.g., range(1, 13) for ECI_1 to ECI_12).

    Returns:
        pd.DataFrame: DataFrame containing yearly correlation values for each ECI_n order.
    """
    # Generate ECI column names based on the given range
    eci_columns = [f"ECI_{n}" for n in n_order]

    # Select relevant columns
    relevant_columns = ["year", "country_id", "log_gdp_per_capita"] + eci_columns
    df = ECI_GDP[relevant_columns].copy()

    # Compute correlation per year for each ECI column
    yearly_correlation = (
        df.set_index(["year", "country_id"])  # Set hierarchical index
        .groupby("year", observed=False)  # Group by year
        .corr()  # Compute correlation matrix
    )

    # Ensure MultiIndex exists
    if isinstance(yearly_correlation.columns, pd.MultiIndex):
        yearly_correlation = yearly_correlation.xs(
            "log_gdp_per_capita", level=1, axis=1
        )  # Extract only log_gdp_per_capita correlations
    else:
        yearly_correlation = yearly_correlation.get(
            "log_gdp_per_capita", yearly_correlation
        )  # Avoid KeyError

    # Convert MultiIndex to wide format
    yearly_correlation = yearly_correlation.unstack()

    # # Rename columns for clarity
    # yearly_correlation.columns = [col for col in yearly_correlation.columns]

    # Drop self-correlation column just before returning
    yearly_correlation = yearly_correlation.drop(
        columns="log_gdp_per_capita", errors="ignore"
    )

    return yearly_correlation.reset_index()


def lineplot_ECI_vs_GDP_correlation(ECI_correlation: pd.DataFrame):
    """
    Generates a line plot to visualize the yearly correlation between Economic Complexity Index (ECI)
    and GDP per capita for even-numbered ECI orders.

    Parameters:
        ECI_correlation (pd.DataFrame):
            A DataFrame containing yearly correlation values between various ECI orders and GDP per capita.
            Expected columns:
                - "year" (int): The year of the correlation measurement.
                - "ECI_n" (float): Correlation values for different ECI orders (e.g., ECI_2, ECI_3, ...).

    Returns:
        None: Displays a Seaborn line plot with multiple lines representing the correlations
              for different even-numbered ECI orders over time.

    Notes:
        ECI is based on the kc measures, even-ordered ECI values reflect diversification, while odd-ordered ECI values reflect ubiquity.
        We select only the even orders (ECI_2, ECI_4, ...), because diversification measures reflect the accumulation of productive capabilities.
    """
    # # Set a clean grid style for better visualization
    # plt.style.use("seaborn-v0_8-darkgrid")

    # Select ECI columns that are even-numbered (excluding "ECI_0")
    eci_columns = [col for col in ECI_correlation.columns if col.startswith("ECI_")]
    even_n_columns = sorted(
        [
            col
            for col in eci_columns
            if int(col.split("_")[1]) % 2 == 0 and col != "ECI_0"
        ],
        key=lambda x: int(x.split("_")[1]),  # Sort by numeric value
    )

    # Melt the DataFrame to long format
    melted_df = ECI_correlation.melt(
        id_vars=["year"],
        value_vars=even_n_columns,
        var_name="ECI",
        value_name="correlation",
    ).dropna()  # Remove missing values to avoid gaps

    y_min = 0.0
    y_max = 0.9

    # Create line plots
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted_df, x="year", y="correlation", hue="ECI", marker="o")
    # sns.lineplot(data=melted_df, x="year", y="correlation", hue="ECI", markers=True, dashes=False)

    # Customize the plot
    plt.title(
        "Yearly Correlation Between ECI and GDP per Capita (Even Orders)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Correlation", fontsize=12)
    # plt.grid(True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(y_min, y_max)

    # plt.legend(title="ECI Columns")
    # Adjust legend to be outside the plot for better readability
    plt.legend(
        title="ECI Order", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10
    )

    plt.show()


def testing_eci_merging():
    from importlib import reload

    # import p03_eci_mor
    # reload(p03_eci_mor)
    # from p03_eci_mor import calculate_RCA, create_M_matrix, n_order_reflections
    import p01_raw_data_imports

    reload(p01_raw_data_imports)
    from p01_raw_data_imports import (
        switch_to_dtype,
        read_hs92_stata_data,
        custom_float_formatter,
        df_stats,
    )

    # Assign the custom formatter to Pandas options
    pd.options.display.float_format = custom_float_formatter

    print("cwd:", os.getcwd(), "=" * 60, sep="\n")

    # Define the relative path to the output data folder
    output_file_location = os.path.join("data", "06_mor_gold")
    print(
        f"output_file_location: {output_file_location}",
        f"exist: {os.path.exists(output_file_location)}",
        "=" * 60,
        sep="\n",
    )

    # Define the relative path to the input data folder
    input_file_location = os.path.join("data", "05_mor_silver")
    print(
        f"input_file_location: {input_file_location}",
        f"exist: {os.path.exists(input_file_location)}",
        "=" * 60,
        sep="\n",
    )

    # Importing info about countries: 'country_code'
    initial_input_location = input_file_location

    input_file_location = os.path.join("data", "01_raw")
    input_file_name = "location_country.csv"
    # reading the csv file, renaming 'iso3_code' to 'country_code'
    location_df = pd.read_csv(
        os.path.join(input_file_location, input_file_name),
        usecols=["country_id", "name_short_en", "iso3_code"],
    )
    location_df.rename(
        columns={"name_short_en": "country_name", "iso3_code": "country_code"},
        inplace=True,
    )

    input_file_location = initial_input_location

    # Switching to the 'category' data type for categorical columns
    categorical_columns = ["country_id", "country_name", "country_code"]
    location_df = switch_to_dtype(
        location_df, default_dtype="str", categorical_columns=categorical_columns
    )

    # Processing ECI data
    input_file_name = "ECI.dta"
    ECI = read_hs92_stata_data(input_file_location, input_file_name)

    # Switching to the 'category' data type for categorical columns
    categorical_columns = ["country_id", "year"]
    ECI = switch_to_dtype(
        ECI, default_dtype="str", categorical_columns=categorical_columns
    )

    # Merging the product_exports_total with location_df to get the country_code
    ECI = ECI.merge(location_df, on="country_id", how="left")

    print("ECI shape:", ECI.shape, "=" * 60, sep="\n")
    # df_stats(ECI)

    # Ordering by year and country_code columns
    ECI.sort_values(by=["year", "country_code"], inplace=True)

    # Processing GDP per capita data
    initial_input_location = input_file_location

    input_file_location = os.path.join("data", "03_preprocessed")
    input_file_name = "GDP.dta"
    categorical_columns = ["country_code", "year"]
    GDP = read_hs92_stata_data(
        input_file_location, input_file_name, categorical_columns=categorical_columns
    )

    input_file_location = initial_input_location

    # Switching to the 'category' data type for categorical columns
    categorical_columns = ["country_code", "year"]
    GDP = switch_to_dtype(
        GDP, default_dtype="str", categorical_columns=categorical_columns
    )

    print("GDP shape:", GDP.shape, "=" * 60, sep="\n")
    # df_stats(GDP)

    ECI_years = set(ECI["year"])
    # subsetting the GDP data to only include the years present in the ECI data
    GDP = GDP[GDP["year"].isin(ECI_years)].reset_index(drop=True)

    print("GDP shape after subsetting to eci-years:", GDP.shape, "=" * 60, sep="\n")
    # df_stats(GDP)

    # Subsetting to countries which are present in both ECI and GDP dataframes
    ECI_countries = set(ECI["country_code"])
    GDP_countries = set(GDP["country_code"])
    print(
        f"ECI_countries: {len(ECI_countries)}",
        f"GDP_countries: {len(GDP_countries)}",
        "=" * 60,
        sep="\n",
    )

    common_clean_countries = ECI_countries.intersection(GDP_countries)
    print(
        f"common ECI-GDP countries:{len(common_clean_countries)};", "=" * 60, sep="\n"
    )

    ECI = ECI[ECI["country_code"].isin(common_clean_countries)]
    print(f"unique ECI countries: {len(set(ECI['country_code']))};", "=" * 60, sep="\n")
    print("ECI shape:", ECI.shape, "=" * 60, sep="\n")

    GDP = GDP[GDP["country_code"].isin(common_clean_countries)]
    print(f"unique GDP countries: {len(set(GDP['country_code']))};", "=" * 60, sep="\n")
    print("GDP shape:", GDP.shape, "=" * 60, sep="\n")

    # Merging ECI to GDP
    year = 2012
    n_order = 12
    ECI_GDP_12_2012 = merge_eci_to_gdp_by_year_v01(ECI, GDP, n=n_order, year=year)
    print("ECI_GDP_12_2012 shape:", ECI_GDP_12_2012.shape, "=" * 60, sep="\n")

    pass


def main(argv=None):
    if argv is None:
        argv = sys.argv

    s = "stop"

    # test comes here
    testing_eci_merging()
    print("=" * 60, "===>>> testing_eci_merging() is done", "=" * 60, sep="\n")

    s = "stop"

    pass


if __name__ == "__main__":
    sys.exit(main())
