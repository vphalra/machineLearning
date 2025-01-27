from dataprocessing import encoded_data
import statsmodels.api as sm

numerical_features = ['Year Built','Bedroom AbvGr', 'Kitchen AbvGr', 'TotalSQFT', 'AdjustedSalePrice', 'TotalBaths', ''
                      'TotRms AbvGrd', 'Fireplaces','Garage Yr Blt','Garage Cars','Wood Deck SF','Open Porch SF','',
                      'Enclosed Porch','3Ssn Porch ','Screen Porch','Pool Area','Misc Val','Mo Sold','Yr Sold',
                      'Year Remod/Add','Mas Vnr Area ','Lot Frontage ','Lot Area','Overall Qual']

encoded_data = encoded_data
print(encoded_data.dtypes)

features_to_drop = ['BsmtFin SF 1', 'BsmtFin SF 2','Bsmt Unf SF', 'Total Bsmt SF','1st Flr SF', '2nd Flr SF',
                    'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath',
                    'Garage Area', 'SalePrice', 'years_to_2024']

data = encoded_data.drop(columns = features_to_drop)
print(data.dtypes)

X = data.drop(columns = ['AdjustedSalePrice'])
y = data['AdjustedSalePrice']
X = sm.add_constant(X)

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

print(data['Wood Deck SF'].describe())