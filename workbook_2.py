from dataprocessing import encoded_data
import statsmodels.api as sm

# List of numerical variables to drop (from your provided list)
numerical_to_drop = [
    'Lot Shape', 'Utilities', 'Land Slope', 'Year Remod/Add', 'Exter Cond',
    'Bsmt Cond', 'BsmtFin Type 2', 'Electrical', 'Kitchen AbvGr', 'Fireplace Qu',
    'Garage Yr Blt', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Paved Drive',
    'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Fence', 'Misc Val', 'Mo Sold'
]

# List of encoded categorical variables to drop
encoded_categorical_to_drop = [
    'MS Zoning_FV', 'MS Zoning_I (all)', 'MS Zoning_RH', 'MS Zoning_RL', 'MS Zoning_RM',
    'Street_Pave',
    'Roof Style_Gable', 'Roof Style_Gambrel', 'Roof Style_Hip', 'Roof Style_Mansard', 'Roof Style_Shed',
    'Alley_NoAlley', 'Alley_Pave',
    'Heating_GasW', 'Heating_Grav', 'Heating_OthW',
    'Misc Feature_None', 'Misc Feature_Othr', 'Misc Feature_Shed', 'Misc Feature_TenC'
]

# List of other numerical features to drop
other_numerical_to_drop = [
    'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF',
    '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath',
    'Full Bath', 'Half Bath', 'Garage Area', 'SalePrice', 'years_to_2024'
]

# Combine all features to drop into a single list
all_features_to_drop = numerical_to_drop + encoded_categorical_to_drop + other_numerical_to_drop

# Drop the specified features from the dataset
data = encoded_data.drop(columns=all_features_to_drop)

# Verify remaining data types after dropping features
print(data.dtypes)

# Define independent variables (X) and dependent variable (y)
X = data.drop(columns=['AdjustedSalePrice'])  # Independent variables
y = data['AdjustedSalePrice']                # Dependent variable

# Add a constant to X for the intercept
X = sm.add_constant(X)

# Fit the OLS regression model
ols_model = sm.OLS(y, X).fit()

# Print the OLS summary
print(ols_model.summary())
