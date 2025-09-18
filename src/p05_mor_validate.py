"""
p04_mor_validation.py
(Economic Complexity Index using Method of Reflections)

This module is a collection of methods used to validate calculated Economic Complexity Index(ECI) using Method of Reflections(MOR)

Purpose:
Provide an ...

Author: Dzmitry Nisht
Date: YYYY-MM-DD
"""

import matplotlib.pyplot as plt
import seaborn as sns
import math

def validate_eci_reflections_big(df, years=None, reflections=None):
    """
    Plot the distribution of Economic Complexity Indices (ECI) for specified years and reflections.

    Args:
        df (pd.DataFrame): DataFrame containing ECI reflections with 'year' column.
        years (list, optional): List of years to include in the plot. Defaults to all available years.
        reflections (list, optional): List of reflections to plot. Defaults to all reflections.
    """
    # Determine available years and reflections
    available_years = df["year"].unique() if years is None else years
    available_reflections = [col for col in df.columns if col.startswith("ECI_")]
    if reflections is not None:
        available_reflections = [
            f"ECI_{n}" for n in reflections if f"ECI_{n}" in df.columns
        ]

    # Plot each year and reflection separately
    for year in available_years:
        df_year = df[df["year"] == year]

        for reflection in available_reflections:
            if reflection in df_year.columns:
                # Count negatives and positives
                neg_count = (df_year[reflection] < 0).sum()
                pos_count = (df_year[reflection] > 0).sum()

                # Plot distribution
                plt.figure(figsize=(8, 5))
                sns.histplot(df_year[reflection], kde=True, bins=30)
                plt.title(f"Distribution of {reflection} for Year {year}")
                plt.xlabel(reflection)
                plt.ylabel("Frequency")

                # Display text with negative/positive counts
                text_str = f"Negatives: {neg_count}\nPositives: {pos_count}"
                plt.text(
                    0.95, 0.95, text_str, transform=plt.gca().transAxes, 
                    fontsize=12, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
                )

                # Show plot
                plt.show()


def validate_eci_reflections_adjusted(df, years=None, reflections=None):
    """
    Plot the distribution of Economic Complexity Indices (ECI) for specified years and reflections.
    
    Args:
        df (pd.DataFrame): DataFrame containing ECI reflections with 'year' column.
        years (list, optional): List of years to include in the plot. Defaults to all available years.
        reflections (list, optional): List of reflections to plot. Defaults to all reflections.
    """
    # Determine available years and reflections
    available_years = df["year"].unique() if years is None else years
    available_reflections = [col for col in df.columns if col.startswith("ECI_")]
    if reflections is not None:
        available_reflections = [f"ECI_{n}" for n in reflections if f"ECI_{n}" in df.columns]
    print("available_reflections", available_reflections, "available_years", available_years)

    # Calculate total number of plots and determine grid size
    total_plots = len(available_years) * len(available_reflections)
    cols = min(4, total_plots)  # Up to 4 plots per row
    rows = math.ceil(total_plots / cols)
    print("total_plots", total_plots, "cols", cols, "rows", rows)
    
    # Adjust figure size dynamically
    fig_width = cols * 5  # Scale width
    fig_height = rows * 4  # Scale height

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if total_plots > 1 else [axes]

    plot_index = 0
    for year in available_years:
        df_year = df[df["year"] == year]

        for reflection in available_reflections:
            if reflection in df_year.columns:
                # Count negatives and positives
                counts = df_year[reflection].apply(lambda x: "Negative" if x < 0 else "Positive").value_counts(normalize=False)
                neg_count = counts.get("Negative", 0)
                pos_count = counts.get("Positive", 0)

                # Count negative and positive shares
                counts = df[reflection].apply(lambda x: "Negative" if x < 0 else "Positive").value_counts(normalize=True)
                neg_share = round(counts.get("Negative", 0) * 100, 1)
                pos_share = round(counts.get("Positive", 0) * 100, 1)

                # Select subplot axis
                ax = axes[plot_index]

                # Plot distribution
                sns.histplot(df_year[reflection], kde=True, bins=30, ax=ax)
                ax.axvline(0, color="red", linestyle="--")  # Mark the zero line
                ax.set_title(f"{reflection} for Year {year}")
                ax.set_xlabel(reflection)
                ax.set_ylabel("Frequency")

                # Display text with negative/positive counts
                text_str = f"Negatives: {neg_count} ({neg_share}%)\nPositives: {pos_count} ({pos_share}%)"
                ax.text(
                    0.95, 0.95, text_str, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
                )

                plot_index += 1

    # Adjust layout
    plt.tight_layout()
    plt.show()