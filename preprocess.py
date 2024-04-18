import pandas as pd

# Let's try to load the CSV file to see what the dataset looks like.
file_path = './data/race_results.csv'

# Load the dataset
race_results = pd.read_csv(file_path)

# Display the first few rows of the dataframe
race_results.head()

# First, we will drop the first row as it contains NaN values across all columns which indicates it might be an error.
race_results_cleaned = race_results.dropna(how='all')

# Checking for the number of missing values in each column
missing_values = race_results_cleaned.isnull().sum()

# For simplicity, let's drop any columns with missing values.
# In a real-world scenario, we might want to impute these or drop only if there's a high percentage of missing data.
race_results_cleaned = race_results_cleaned.dropna(axis=1)

# Checking the data types of the remaining columns to ensure they are numerical (suitable for regression models)
data_types = race_results_cleaned.dtypes

# We also want to make sure that the 'Finish Time (minutes)' is in the dataframe since it's our target variable.
# If it has been dropped due to missing values, we need to reconsider our strategy.
finish_time_included = 'Finish Time (minutes)' in race_results_cleaned.columns