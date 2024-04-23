import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

# Assuming your dataframe is named `race_results_cleaned` and has a column 'Start Time' with time strings
def time_to_float(time_string):
    # Parse time string to a datetime object
    time_obj = datetime.strptime(time_string, '%I:%M %p')
    # Convert to total hours as a float
    return time_obj.hour + time_obj.minute / 60

# Let's try to load the CSV file to see what the dataset looks like.
file_path = './data/race_results_4_18_2024.csv'

# Load the dataset
race_results = pd.read_csv(file_path)

# Display the first few rows of the dataframe
race_results.head()

encoder = OneHotEncoder()
teaching_encoded = encoder.fit_transform(race_results[['Teaching?']]).toarray()
race_results = race_results.join(pd.DataFrame(teaching_encoded, columns=encoder.get_feature_names_out()))

# 로그 변환 및 스케일링 for 'Age of youngest child(days)'
scaler = StandardScaler()
race_results['Age of youngest child (Days)'] = np.log1p(race_results['Age of youngest child (Days)'])
race_results['Age of youngest child (Days)'] = scaler.fit_transform(race_results[['Age of youngest child (Days)']])


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

# Drop 'Race Name' and 'Start Time' columns as they are not likely to be useful for our model
#race_results_cleaned = race_results_cleaned.drop(['Race Name', 'Start Time'], axis=1)

# Convert 'Date' to datetime and create a numerical feature from it (e.g., days since the first race in the dataset)
race_results_cleaned['Date'] = pd.to_datetime(race_results_cleaned['Date'])
race_results_cleaned['Days Since First Race'] = (race_results_cleaned['Date'] - race_results_cleaned['Date'].min()).dt.days

# One-hot encode the 'Condition' categorical variable
# First, we instantiate the encoder
encoder = OneHotEncoder(drop='first')

# Perform one-hot encoding and convert to a dataframe
encoded_conditions = encoder.fit_transform(race_results_cleaned[['Condition']]).toarray()
encoded_conditions_df = pd.DataFrame(encoded_conditions, columns=encoder.get_feature_names_out(['Condition']))

# Join the encoded dataframe with the original one and drop the original 'Condition' column
race_results_cleaned = race_results_cleaned.join(encoded_conditions_df).drop('Condition', axis=1)
race_results_cleaned['Start Time Float'] = race_results_cleaned['Start Time'].apply(time_to_float)

# Calculate pace: finish time per mile
race_results_cleaned['Pace per Mile'] = race_results_cleaned['Finish Time (minutes)'] / race_results_cleaned['Distance (miles)']

race_results_cleaned = race_results_cleaned.drop(['Race Name', 'Date', 'Start Time', 'Finish Time (minutes)', 'Teaching?', 'Age of youngest child (Days)', 'Teaching?_No','Teaching?_Yes'], axis=1)

# 데이터 프레임을 CSV 파일로 저장
race_results_cleaned.to_csv('processed_race_results.csv', index=False)

numerical_features = race_results_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
for column in numerical_features:
    race_results_cleaned[column] = race_results_cleaned[column].apply(lambda x: np.log(x+1))

x_test = race_results_cleaned.iloc[-1]
y_test = x_test['Pace per Mile']
x_test = x_test.drop(['Pace per Mile'])
race_results.drop(race_results_cleaned.index[-1], inplace=True)

print(x_test, y_test)

# Define the new target variable y as 'Pace per Mile'
y = race_results_cleaned['Pace per Mile']

# Drop the original 'Finish Time (minutes)' and 'Pace per Mile' from the features
X = race_results_cleaned.drop(['Pace per Mile'], axis=1)


#X = race_results_cleaned.iloc[:, :-1]  # All columns except the last one as features
#y = race_results_cleaned.iloc[:, -1]   # The last column as the target variable

# Reset the index to ensure proper alignment
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# Separate the last row for testing
X, x_test = X.iloc[:-1], X.iloc[-1:]
y, y_test = y.iloc[:-1], y.iloc[-1]

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the scaler
scaler = StandardScaler()


# Initialize the Random Forest Regressor
rf_reg_model = RandomForestRegressor(random_state=42)
# Fit the model to the training data
rf_reg_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = rf_reg_model.predict(X_val)
print(X_val)
print(y_val)
print(y_val_pred)

# Calculate performance metrics
rf_mse = mean_squared_error(y_val, y_val_pred)
rf_mae = mean_absolute_error(y_val, y_val_pred)
rf_r2 = r2_score(y_val, y_val_pred)

print('Random Forest Regressor Mean Squared Error:', rf_mse)
print('Random Forest Regressor Mean Absolute Error:', rf_mae)
print('Random Forest Regressor R^2 Score:', rf_r2)

# x_test에 대한 예측
predicted_finish_time = rf_reg_model.predict(x_test)
print('Predicted Finish Time for last entry:', predicted_finish_time)
original_value = np.exp(predicted_finish_time) - 1
print(3.1*original_value)
#print(3.1*(np.exp(1.948212)-1))

# For confidence intervals, you can use the bootstrapping method:
import seaborn as sns

# Bootstrap your predictions to get a distribution
predictions = [rf_reg_model.predict(x_test.values.reshape(1, -1)) for _ in range(1000)]

# Calculate the confidence intervals from the distribution of predictions
ci_lower = np.percentile(predictions, 2.5)
ci_upper = np.percentile(predictions, 97.5)

print(f'95% confidence interval for the predictions: [{ci_lower}, {ci_upper}]')