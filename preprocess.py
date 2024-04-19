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

# Base models
estimators = [
    ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), SVR())),
    ('dt', DecisionTreeRegressor())
]

# Stacking ensemble
stack_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

# Assuming your dataframe is named `race_results_cleaned` and has a column 'Start Time' with time strings
def time_to_float(time_string):
    # Parse time string to a datetime object
    time_obj = datetime.strptime(time_string, '%I:%M %p')
    # Convert to total hours as a float
    return time_obj.hour + time_obj.minute / 60

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

race_results_cleaned = race_results_cleaned.drop(['Race Name', 'Date', 'Start Time', 'Finish Time (minutes)'], axis=1)

# Define the new target variable y as 'Pace per Mile'
y = race_results_cleaned['Pace per Mile']

# Drop the original 'Finish Time (minutes)' and 'Pace per Mile' from the features
X = race_results_cleaned.drop(['Pace per Mile'], axis=1)

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to the validation data
X_val_scaled = scaler.transform(X_val)


selector = SelectKBest(score_func=f_regression, k=10)  # 'k'는 선택할 피처의 수

# Fit the selector to the training data and transform it
X_train_selected = selector.fit_transform(X_train, y_train)

# Apply the same transformation to the validation data
X_val_selected = selector.transform(X_val)

# Print the selected feature indices
selected_features_indices = selector.get_support(indices=True)
selected_feature_names = X_train.columns[selected_features_indices]
print("Selected features:", selected_feature_names)



# Initialize the Linear Regression model
lin_reg_model = LinearRegression()

# Fit the model to the training data
lin_reg_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = lin_reg_model.predict(X_val)

# Calculate performance metrics
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)
print('R^2 Score:', r2)

# Initialize the Random Forest Regressor
rf_reg_model = RandomForestRegressor(random_state=42)
sfm = SelectFromModel(estimator=rf_reg_model)

# Fit the model to the training data and transform it
X_train_selected = sfm.fit_transform(X_train_scaled, y_train)

# Apply the same transformation to the validation data
X_val_selected = sfm.transform(X_val_scaled)

# Find the features that were selected
selected_features_indices = sfm.get_support(indices=True)
selected_feature_names = X_train.columns[selected_features_indices]
print("Selected features:", selected_feature_names)

# # Fit the model to the training data
# rf_reg_model.fit(X_train, y_train)

# # Predict on the validation set
# y_val_pred = rf_reg_model.predict(X_val)

# # Calculate performance metrics
# rf_mse = mean_squared_error(y_val, y_val_pred)
# rf_mae = mean_absolute_error(y_val, y_val_pred)
# rf_r2 = r2_score(y_val, y_val_pred)

# print('Random Forest Regressor Mean Squared Error:', rf_mse)
# print('Random Forest Regressor Mean Absolute Error:', rf_mae)
# print('Random Forest Regressor R^2 Score:', rf_r2)


# Train the model using the scaled data
rf_reg_model.fit(X_train_selected, y_train)

# Predict on the scaled validation set
y_val_pred = rf_reg_model.predict(X_val_selected)

# Calculate performance metrics
rf_mse_scaled = mean_squared_error(y_val, y_val_pred)
rf_mae_scaled = mean_absolute_error(y_val, y_val_pred)
rf_r2_scaled = r2_score(y_val, y_val_pred)

print('Random Forest Regressor with Scaled Data - Mean Squared Error:', rf_mse_scaled)
print('Random Forest Regressor with Scaled Data - Mean Absolute Error:', rf_mae_scaled)
print('Random Forest Regressor with Scaled Data - R^2 Score:', rf_r2_scaled)


# Fit the stacking ensemble
stack_reg.fit(X_train_scaled, y_train)

# Predict on the validation set
y_val_pred = stack_reg.predict(X_val_scaled)

# Calculate performance metrics
stack_mse = mean_squared_error(y_val, y_val_pred)
stack_mae = mean_absolute_error(y_val, y_val_pred)
stack_r2 = r2_score(y_val, y_val_pred)

print('Stacking Regressor Mean Squared Error:', stack_mse)
print('Stacking Regressor Mean Absolute Error:', stack_mae)
print('Stacking Regressor R^2 Score:', stack_r2)