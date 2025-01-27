from dataprocessing import encoded_data
import statsmodels.api as sm

# List of numerical variables to retain (only numerical)
numerical_vars_to_keep = [
    'Year Built', 'Bedroom AbvGr', 'Kitchen AbvGr', 'TotalSQFT', 'AdjustedSalePrice',
    'TotalBaths', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars',
    'Wood Deck SF', 'Screen Porch',
     'Yr Sold', 'Year Remod/Add', 'Mas Vnr Area',
    'Lot Frontage', 'Lot Area', 'Overall Qual'
]

# Filter the dataset to include only the specified numerical variables
numerical_data = encoded_data[numerical_vars_to_keep]

# Verify the data types to ensure only numerical columns remain
print(numerical_data.dtypes)

# Define independent variables (X) and dependent variable (y)
X = numerical_data.drop(columns=['AdjustedSalePrice'])  # Independent variables
y = numerical_data['AdjustedSalePrice']                # Dependent variable

# Add a constant to X for the intercept
X = sm.add_constant(X)

# Fit the OLS regression model
ols_model = sm.OLS(y, X).fit()

# Print the OLS summary
print(ols_model.summary())
