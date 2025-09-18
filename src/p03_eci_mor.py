"""
eci_mor.py
(Economic Complexity Index using Method of Reflections)

This module is a collection of methods used to calculate Economic Complexity Index(ECI) using Method of Reflections(MOR)

Purpose:
Provide an ...

Author: Dzmitry Nisht
Date: YYYY-MM-DD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from p01_raw_data_imports import df_stats


# Calculate Revealed Comparative Advantage (RCA) using numpy arrays, broadcasting, and 1 year of synthetic data
def calculate_np_1_year_RCA(x_cp):
    """
    Calculate Revealed Comparative Advantage (RCA).
    To make countries and products more readily comparable, we control for variations in the size of countries and of product markets,
    as the ratio between the export share of product p in country c and the share of product p in the world market,
    according to formula:
    RCA_cp = (x_cp/Σp x_cp)/(Σc x_cp/Σc Σp x_cp)
    where x_cp represents the dollar exports of country c in product p.
    Args:
        x_cp
    Returns:
        RCA
        sum_products
        sum_countries
    """
    # Calculate denominators: Σp x_cp (sum over products)
    product_totals = np.sum(x_cp, axis=1, keepdims=True)

    # Calculate Σc x_cp (sum over countries)
    country_totals = np.sum(x_cp, axis=0, keepdims=True)

    # Calculate total sum Σc Σp x_cp
    total_exports = np.sum(x_cp)

    # Apply RCA formula
    RCA = (x_cp / product_totals) / (country_totals / total_exports)

    return RCA, product_totals, country_totals


# copilot
def calculate_RCA(x_cp):
    """
    Calculate the Revealed Comparative Advantage (RCA).
    To make countries and products more readily comparable, we control for variations in the size of countries and of product markets,
    as the ratio between the export share of product p in country c and the share of product p in the world market,
    according to formula:
    RCA_cp = (x_cp/Σp x_cp)/(Σc x_cp/Σc Σp x_cp)
    where x_cp represents the dollar exports of country c in product p across multiple years.
    Args:
        x_cp: DataFrame with export data, which can be spread across multiple years.
            Should contain columns: 'country_id', 'product_id', 'average_export_value', 'year'.
    Returns:
        RCA: x_cp DataFrame with the RCA values calculated for each country-product pair in each year,
            as well as the total exports by country 'country_total' and product 'product_total'
    """

    # Group by year and calculate RCA for each year
    def calculate_RCA_for_year(group):
        # Total exports by country
        country_totals = group.groupby("country_id", observed=False)[
            "average_export_value"
        ].sum()

        # Total exports by product
        product_totals = group.groupby("product_id", observed=False)[
            "average_export_value"
        ].sum()

        # Total exports overall
        total_exports = group["average_export_value"].sum()

        # Merge the totals back to the original DataFrame
        group = group.merge(country_totals.rename("country_total"), on="country_id")
        group = group.merge(product_totals.rename("product_total"), on="product_id")

        # Calculate RCA
        group["RCA"] = (group["average_export_value"] / group["country_total"]) / (
            group["product_total"] / total_exports
        )

        return group

    # Apply the RCA calculation for each year
    RCA = (
        x_cp.groupby("year", observed=False)
        .apply(calculate_RCA_for_year)
        .reset_index(drop=True)
    )

    return RCA


def create_np_M_matrix(RCA):
    """
    Define the Mcp binary adjacency matrix, summarizing the connections between countries and the products they export where:
    M_cp = 1 if RCA >= 1
    M_cp = 0 otherwise
    """
    return (RCA >= 1).astype(int)


def create_M_matrix(RCA):
    """
    Define the Mcp binary adjacency matrix, summarizing the connections between countries and the products they export where:
    M_cp = 1 if RCA >= 1
    M_cp = 0 otherwise
    """
    # Create a copy of the DataFrame to avoid the SettingWithCopyWarning
    RCA_copy = RCA.copy()

    # Create the 'M' column based on the condition
    RCA_copy.loc[:, "M"] = (RCA_copy["RCA"] >= 1).astype(int)

    # Select the relevant columns to create the binary matrix M_cp
    M_cp = RCA_copy[["country_id", "product_id", "year", "M"]]

    return M_cp


def initialize_reflections(M_cp, order=1):
    # k_cn, k_pn, ECI, PCI = None, None, None, None

    # Number of iterations for recursion
    n_iterations = order

    # Initialize arrays for k_cn and k_pn
    k_cn = np.zeros((M_cp.shape[0], n_iterations + 1))
    k_pn = np.zeros((M_cp.shape[1], n_iterations + 1))
    print(
        f"Initialized k_cn (Reflections of countries diversity across iterations):\n{k_cn}\n"
    )
    print(
        f"Initialized k_pn (Reflections of products ubiquity across iterations):\n{k_pn}\n"
    )

    # Set initial values ECI and PCI
    ECI = np.zeros(k_cn.shape)
    PCI = np.zeros(k_pn.shape)

    return k_cn, k_pn, ECI, PCI


def n_order_reflections_np(
    M_cp=None, k_cn=None, k_pn=None, ECI=None, PCI=None, order=1
):
    """Improved version of n_order_reflections with eci calculated"""

    # Step 3: Diversity
    # We calculate the diversification of country c as the sum of Mcp across all products (axis=1) k_c0
    k_c0 = np.sum(M_cp, axis=1)
    # Step 4: Ubiquity
    # We calculate the ubiquity of product p as the sum of Mcp across all countries (axis=0) k_p0
    k_p0 = np.sum(M_cp, axis=0)

    # Conditional Indexing: Instead of using loops to check conditions like division by zero, the code uses boolean indexing (valid_k_c0 and valid_k_p0) to apply operations only where valid.
    valid_k_c0 = k_c0 != 0
    valid_k_p0 = k_p0 != 0

    # Number of iterations for recursion
    n_iterations = order

    # Set initial values (order=0) of diversity(k_cn) and ubiquity(k_pn)
    order_0 = 0
    k_cn[:, order_0] = k_c0
    k_pn[:, order_0] = k_p0
    print(f"k_c0 (Diversity of countries):\n{k_cn}\n")
    print(f"k_p0 (Ubiquity of products):\n{k_pn}\n")

    std_k_c0 = np.std(k_cn[valid_k_c0, order_0])
    deviation_c0 = k_cn[valid_k_c0, order_0] - np.mean(k_cn[valid_k_c0, order_0])
    ECI[valid_k_c0, order_0] = (
        deviation_c0 if std_k_c0 == 0 else deviation_c0 / std_k_c0
    )

    std_k_p0 = np.std(k_pn[valid_k_p0, order_0])
    deviation_p0 = k_pn[valid_k_p0, order_0] - np.mean(k_pn[valid_k_p0, order_0])
    PCI[valid_k_p0, order_0] = (
        deviation_p0 if std_k_p0 == 0 else deviation_p0 / std_k_p0
    )

    # Recursive calculation of k_cn and k_pn for each iteration
    # Vectorization: The code now uses np.dot for matrix multiplication instead of loops, which is more efficient and takes advantage of NumPy's optimized operations.
    for n in range(n_iterations):
        # Update measure of average ubiquity of the products k_cn for n+1 order using vectorized operations
        k_cn[valid_k_c0, n + 1] = (
            np.dot(M_cp[valid_k_c0], k_pn[:, n]) / k_c0[valid_k_c0]
        )

        std_k_cn = np.std(k_cn[valid_k_c0, n + 1])
        deviation_cn = k_cn[valid_k_c0, n + 1] - np.mean(k_cn[valid_k_c0, n + 1])
        ECI[valid_k_c0, n + 1] = (
            deviation_cn if std_k_cn == 0 else deviation_cn / std_k_cn
        )

        # Update measure of average diversity of the countries k_pn for n+1 order using vectorized operations
        k_pn[valid_k_p0, n + 1] = (
            np.dot(M_cp[:, valid_k_p0].T, k_cn[:, n]) / k_p0[valid_k_p0]
        )

        std_k_pn = np.std(k_pn[valid_k_p0, n + 1])
        deviation_pn = k_pn[valid_k_p0, n + 1] - np.mean(k_pn[valid_k_p0, n + 1])
        PCI[valid_k_p0, n + 1] = (
            deviation_pn if std_k_pn == 0 else deviation_pn / std_k_pn
        )

    return k_c0, k_cn, k_p0, k_pn, ECI, PCI


def calculate_ubiquity_diversity(group):
    """
    Calculate the diversity (k_c0) and ubiquity (k_p0).
    """
    # Calculate diversity of country c as a sum of M_cp across all products p
    diversity = group.groupby("country_id", observed=False)["M"].sum().rename("k_c0")

    # Calculate ubiquity of product p as a sum of M_cp across all countries c
    ubiquity = group.groupby("product_id", observed=False)["M"].sum().rename("k_p0")

    # Merge the results back into the original group
    group = group.merge(diversity, on="country_id")
    group = group.merge(ubiquity, on="product_id")

    return group


def n_order_reflections(M_cp=None, order=1):
    """
    Calculate n-order reflections for the M_cp DataFrame.
    The DataFrame should contain columns: 'country_id', 'product_id', 'year', 'M'.
    """
    if M_cp is None:
        raise ValueError("M_cp DataFrame cannot be None")

    # Calculate the diversity (k_c0) and ubiquity (k_p0) for each year,
    #   applying the calculate_ubiquity_diversity function for each year
    M_cp = (
        M_cp.groupby("year", observed=False)
        .apply(calculate_ubiquity_diversity)
        .reset_index(drop=True)
    )

    # Filter out rows where k_c0 or k_p0 are zero
    M_cp = M_cp[(M_cp["k_c0"] != 0) & (M_cp["k_p0"] != 0)]

    # Calculate standardized values for ECI and PCI
    M_cp["ECI_0"] = M_cp.groupby("year", observed=False)["k_c0"].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
    )

    M_cp["PCI_0"] = M_cp.groupby("year", observed=False)["k_p0"].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
    )

    # Initialize k_cn and k_pn columns for each order of reflection
    for n in range(order):
        # Update k_cn measure of average Ubiquity of the products for n+1 order of reflection
        M_cp["k_c" + str(n + 1)] = M_cp.groupby(["year", "country_id"], observed=False)[
            "M"
        ].transform(
            lambda x: (x * M_cp.loc[x.index, "k_p" + str(n)]).sum()
            / M_cp.loc[x.index, "k_c0"].iloc[0]
        )

        # Update k_pn measure of average Diversity of the countries for n+1 order of reflection
        M_cp["k_p" + str(n + 1)] = M_cp.groupby(["year", "product_id"], observed=False)[
            "M"
        ].transform(
            lambda x: (x * M_cp.loc[x.index, "k_c" + str(n)]).sum()
            / M_cp.loc[x.index, "k_p0"].iloc[0]
        )

        # Calculate standardized values for ECI and PCI
        M_cp["ECI_" + str(n + 1)] = M_cp.groupby("year", observed=False)[
            "k_c" + str(n + 1)
        ].transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))

        M_cp["PCI_" + str(n + 1)] = M_cp.groupby("year", observed=False)[
            "k_p" + str(n + 1)
        ].transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1))

    # At this point we have all calculated values for ECI and PCI for each order of reflection in the M_cp DataFrame
    # We can now return the DataFrame or extract the values as needed
    df_stats(M_cp, "M_cp")

    # Assuming that we want ECI and PCI as a standalone DataFrames for each country and product
    #   I think it might be handy to have a diversity index for ECIs and ubiquity index for PCIs
    ECI = M_cp[
        ["year", "country_id", "k_c0", *["ECI_" + str(n) for n in range(order + 1)]]
    ]
    PCI = M_cp[
        ["year", "product_id", "k_p0", *["PCI_" + str(n) for n in range(order + 1)]]
    ]

    # Assuming that we want K_c and K_p reflections in a standalone DataFrames
    K_c = M_cp[["year", "country_id", *["k_c" + str(n) for n in range(order + 1)]]]
    K_p = M_cp[["year", "product_id", *["k_p" + str(n) for n in range(order + 1)]]]

    # # Subsetting the M_cp DataFrame to exclude all ECI, PCI, K_c and K_p columns, leaving only the binary matrix M_cp, diversity and ubiquity columns
    M_cp = M_cp[["year", "country_id", "product_id", "M", "k_c0", "k_p0"]]

    return M_cp, ECI, PCI, K_c, K_p


def estimate_average_sparsity(
    df: pd.DataFrame, sparsity_col_name: str = "average_sparsity"
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calculate sparsity for each year separately, rather than across all years.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'year', 'product_id', and 'country_id' columns.
    sparsity_col_name (str, optional): Name of the sparsity column in the output DataFrame. Defaults to "average_sparsity".

    Returns:
    tuple[pd.Series, pd.DataFrame]:
        - Series with the count of unique countries exporting each product per year.
        - DataFrame summarizing the average sparsity per year.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input is not a DataFrame")

    # Step 1: Count the number of countries exporting each product per year
    product_country_counts = df.groupby(["year", "product_id"], observed=False)[
        "country_id"
    ].nunique()

    # Step 2: Count total number of countries in each year
    total_countries_per_year = df.groupby("year", observed=False)[
        "country_id"
    ].nunique()

    # Step 3: Compute yearly sparsity (percentage of zeros per product per year)
    sparsity_per_year = 1 - (product_country_counts / total_countries_per_year)

    # Step 4: Summarize sparsity levels per year
    sparsity_summary = (
        sparsity_per_year.groupby("year", observed=False).mean().reset_index()
    )
    sparsity_summary.columns = ["year", sparsity_col_name]

    return product_country_counts, sparsity_summary


def plot_product_ubiquity(df: pd.DataFrame, product_country_counts: pd.Series):
    # Plot Yearly Histograms of Product Ubiquity
    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(12, 10), sharex=True, sharey=True
    )
    axes = axes.flatten()

    years = sorted(df["year"].unique())

    for i, year in enumerate(years[:9]):  # Display up to 9 years
        ax = axes[i]
        product_ubiquity = product_country_counts[
            product_country_counts.index.get_level_values("year") == year
        ]

        ax.hist(product_ubiquity, bins=50, edgecolor="black", alpha=0.7)
        ax.set_title(f"Product Ubiquity - Year {year}")
        ax.set_xlabel("Number of Countries Exporting a Product")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_product_ubiquity(df: pd.DataFrame, product_country_counts: pd.Series):
    """
    Plot yearly histograms of product ubiquity using Seaborn.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'year' column.
    product_country_counts (pd.Series): Series containing the count of unique countries exporting each product per year.

    Displays:
    A grid of histograms showing the distribution of product ubiquity over multiple years.
    """
    years = sorted(df["year"].unique())
    num_years = min(9, len(years))  # Display up to 9 years

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(12, 10), sharex=True, sharey=True
    )
    axes = axes.flatten()

    for i, year in enumerate(years[:num_years]):
        ax = axes[i]
        product_ubiquity = product_country_counts[
            product_country_counts.index.get_level_values("year") == year
        ]

        sns.histplot(product_ubiquity, bins=50, kde=True, ax=ax, color="blue")
        ax.set_title(f"Product Ubiquity - Year {year}")
        ax.set_xlabel("Number of Countries Exporting a Product")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
