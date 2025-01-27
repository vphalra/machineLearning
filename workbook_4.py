from dataprocessing import encoded_data
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# List of numerical variables to retain (only numerical)
numerical_vars_to_keep = [
    'Year Built', 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotalSQFT', 'AdjustedSalePrice',
    'TotalBaths', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars',
    'Screen Porch', 'Year Remod/Add', 'Mas Vnr Area', 'Lot Area', 'Overall Qual'
]

# Filter the dataset to include only the specified numerical variables
numerical_data = encoded_data[numerical_vars_to_keep]

# Split the data into train and test sets
X = numerical_data.drop(columns=['AdjustedSalePrice'])  # Independent variables
y = numerical_data['AdjustedSalePrice']                # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to X for OLS
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)

# Store model results
results = {}

# 1. Ordinary Least Squares (OLS) Model
ols_model = sm.OLS(y_train, X_train_ols).fit()
y_pred_ols = ols_model.predict(X_test_ols)
results['OLS'] = {
    'R2': r2_score(y_test, y_pred_ols),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ols)),
    'Model': ols_model
}
print("OLS Summary:")
print(ols_model.summary())

# 2. Linear Regression (using sklearn)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
results['LinearRegression'] = {
    'R2': r2_score(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr))
}

# 3. Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
results['Ridge'] = {
    'R2': r2_score(y_test, y_pred_ridge),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge))
}

# 4. Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
results['Lasso'] = {
    'R2': r2_score(y_test, y_pred_lasso),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lasso))
}

# 5. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
results['RandomForest'] = {
    'R2': r2_score(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf))
}

# 6. XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
results['XGBoost'] = {
    'R2': r2_score(y_test, y_pred_xgb),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb))
}

# Print Results
print("\nModel Performance:")
for model_name, metrics in results.items():
    print(f"{model_name}: R2 = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")

import joblib as jl
jl.dump(xgb_model, '../xgb_model.joblib')