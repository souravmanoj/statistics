# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:17:34 2023

@author: soura
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def read_data_file(file_path):
    """
    Reads a data file in CSV format and returns a Pandas DataFrame.

    Parameters:
    file_path (str): The path to the data file.

    Returns:
    pandas.DataFrame: The data read from the file.
    """
    data = pd.read_csv(file_path, skiprows=4)
    data = data.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
    data = data.set_index('Country Name')
    return data

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("C:\\Users\\soura\\OneDrive\\Desktop\\data2\\co2.csv", encoding='latin-1', skiprows=4)

# Set the index of the DataFrame to the "Country Name" column
df = df.set_index("Country Name")

# Drop unnecessary columns
df = df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1)

# Select only the 12 countries of interest
countries = ["United States", "China", "India", "Netherlands", "Japan", "Germany", "Canada", "Brazil", "Australia", "United Kingdom", "France", "South africa"]
df = df.loc[countries]

# Select only the years of interest with 5 years as increment
years = [str(i) for i in range(1990, 2015, 5)]
df = df[years]

# Transpose the DataFrame to make the countries the index
df = df.transpose()

# Create a bar plot of the CO2 emissions for each country
ax = df.plot(kind="bar", title="CO2 Emissions by Country", xlabel="Year", ylabel="CO2 Emissions (metric tons per capita)")

# Move the legend outside the box
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# Compute the correlation matrix
corr_matrix = df.corr()

# Show the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# Print descriptive statistics
print("Descriptive Statistics:")
print(df.describe())

# Compute skewness and kurtosis
print("Skewness:")
print(skew(df))
print("Kurtosis:")
print(kurtosis(df))


def load_data(file_path):
    """
    Reads the CSV file using pandas and returns a cleaned and transposed DataFrame.
    
    Args:
    file_path (str): The path to the CSV file.
    
    Returns:
    pandas.DataFrame: A cleaned and transposed DataFrame.
    """
    df = pd.read_csv(file_path, encoding='latin-1', skiprows=4)
    df = df.set_index("Country Name")
    df = df.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1)
    countries = ["United States", "China", "India", "Netherlands", "Japan", "Germany", "Canada", "Brazil", "Australia", "United Kingdom", "France", "South Africa"]
    df = df.loc[countries]
    df = df.loc[:, '1990':'2015':5]
    df = df.transpose()
    return df

def plot_data(df):
    """
    Plots the greenhouse gas emissions for each country using a bar plot.
    
    Args:
    df (pandas.DataFrame): A cleaned and transposed DataFrame.
    
    Returns:
    None
    """
    ax = df.plot(kind="bar", title="Greenhouse Gas Emissions by Country", ylabel="Greenhouse Gas Emissions (kt of CO2 equivalent)")
    tick_locs = np.linspace(0, df.shape[0]-1, num=len(df.index))
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(df.index, rotation=45)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.set_ylabel("Greenhouse Gas Emissions (kt of CO2 equivalent)", rotation=270, labelpad=15)
    plt.show()

if __name__ == "__main__":
    file_path = "C:\\Users\\soura\\OneDrive\\Desktop\\data2\\green house.csv"
    df = load_data(file_path)
    print(df.describe())
    plot_data(df)


def plot_methane_emissions(data):
    """
    This function takes a pandas DataFrame containing methane emissions data for various countries and
    plots a line graph of the emissions for each country over time.
    :param data: a pandas DataFrame containing methane emissions data
    """
    # Set the index of the DataFrame to the "Country Name" column
    data = data.set_index("Country Name")

    # Drop unnecessary columns
    data = data.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1)

    # Select only the 13 countries of interest
    countries = ["United States", "China", "India", "Netherlands", "Brazil", "Australia", "Canada", "Argentina", "Germany", "United Kingdom", "Italy", "France", "Japan"]
    data = data.loc[countries]

    # Get only the columns from 1990 to 2020 with 5-year increments
    data = data.loc[:, '1990':'2015':5]

    # Transpose the DataFrame to make the years the index
    data_transposed = data.transpose()

    # Use the .describe() method to get summary statistics for the data
    print("Summary statistics for methane emissions data:\n")
    print(data_transposed.describe())

    # Create a line plot of the methane emissions for each country
    ax = data_transposed.plot(kind="line", title="Methane Emissions by Country", ylabel="Methane Emissions (kt of CO2 equivalent)")

    # Generate evenly spaced tick locations for the x-axis
    tick_locs = list(range(len(data_transposed.index)))

    # Set the x-axis tick locations and labels
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(data_transposed.index, rotation=45)

    # Create a y-axis label on the right side of the plot
    ax.set_ylabel("Methane Emissions (kt of CO2 equivalent)", rotation=270, labelpad=15)

    # Move the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show the plot
    plt.show()


# Read the CSV file into a pandas DataFrame
emissions_data = pd.read_csv("C:\\Users\\soura\\OneDrive\\Desktop\\data2\\methane.csv", encoding='latin-1', skiprows=4)

# Call the plot_methane_emissions function with the DataFrame as the argument
plot_methane_emissions(emissions_data)


def plot_nitrous_oxide_emissions(data):
    """
    This function takes a pandas DataFrame containing nitrous oxide emissions data for various countries and
    plots a line graph of the emissions for each country over time.
    :param data: a pandas DataFrame containing nitrous oxide emissions data
    """
    # Set the index of the DataFrame to the "Country Name" column
    data = data.set_index("Country Name")

    # Drop unnecessary columns
    data = data.drop(["Country Code", "Indicator Name", "Indicator Code"], axis=1)

    # Select only the 13 countries of interest
    countries = ["United States", "China", "India", "Netherlands", "Japan", "Germany", "Canada", "Brazil", "Australia", "United Kingdom", "France", "Argentina", "Italy"]
    data = data.loc[countries]

    # Get only the columns from 1990 to 2015 with 5-year increments
    data = data.loc[:, '1990':'2015':5]

    # Transpose the DataFrame for easier plotting
    data_transposed = data.transpose()

    # Use the .describe() method to get summary statistics for the data
    print("Summary statistics for nitrous oxide emissions data:\n")
    print(data_transposed.describe())

    # Create a line plot of the nitrous oxide emissions for each country
    ax = data_transposed.plot(kind="line", title="Nitrous Oxide Emissions by Country", ylabel="Nitrous Oxide Emissions (kt of CO2 equivalent)")

    # Set the x-axis tick locations and labels
    tick_locs = np.arange(len(data.columns))
    tick_labels = data.columns
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)

    # Create a legend outside the box
    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left")

    # Show the plot
    plt.show()

# Read the CSV file into a pandas DataFrame
emissions_data = pd.read_csv("C:\\Users\\soura\\OneDrive\\Desktop\\data2\\nitrous.csv", encoding='latin-1', skiprows=4)

# Call the plot_nitrous_oxide_emissions function with the DataFrame as the argument
plot_nitrous_oxide_emissions(emissions_data)


