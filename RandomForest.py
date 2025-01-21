"""
The purpose of this python file is to produce a price predictive model
using reduced & statistically significant variables
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dataprocessing import data


# Select features
selected_features = ['Lot Area', 'TotalSQFT', 'TotalBaths', 'Garage Cars', 'Year Built',
                     'Bedroom AbvGr', 'Kitchen AbvGr', 'Overall Qual', 'AdjustedSalePrice']

# Filter data with selected features
reduced_data = data[selected_features]

# Data Splitting
x = reduced_data.drop(columns='AdjustedSalePrice')
y = reduced_data['AdjustedSalePrice']

# Correct train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=420)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=420)
rf_model.fit(x_train, y_train)

# Evaluate Random Forest Model
y_pred_rf = rf_model.predict(x_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"\nRandom Forest Regressor - RMSE: {mse_rf ** 0.5:.2f}")
print(f"Random Forest Regressor - R² Score: {r2_rf:.2f}")

from joblib import dump
dump(rf_model, '../RandomForestPricePredictor.joblib')

# Prediction on Zillow Data
zillow_data = {
    'Lot Area': 1176.2,
    'TotalSQFT': 2000,
    'TotalBaths': 3,
    'Garage Cars': 2,
    'Year Built': 2000,
    'Bedroom AbvGr': 2,
    'Kitchen AbvGr': 1,
    'Overall Qual': 10
}

# Convert Zillow data to DataFrame
zillow_df = pd.DataFrame([zillow_data])

# Ensure alignment with the model's features
zillow_df = zillow_df.reindex(columns=x_train.columns, fill_value=0)

aligned_df = zillow_df

zillow_prediction = rf_model.predict(aligned_df)
print(zillow_prediction)




