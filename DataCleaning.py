"""
This python file's objective is to compound SalePrice's present value to 2024's future value
and clean the data for model training
"""

from dataPrep.GrowthRate import growth_rate
from dataPrep.GrowthRate import data

# Step 1: Adjust SalePrice to 2024's future value
data['years_to_2024'] = 2024 - data['Yr Sold']

data['AdjustedSalePrice'] = data.apply(
    lambda row: row['SalePrice'] * (1 + growth_rate) ** row['years_to_2024'], axis=1
)

# Step 2: Filter Data by Sale Condition
data = data[data['Sale Condition'] == 'Normal']

# Impute missing values
data['Pool QC'] = data.apply(
    lambda row: 'NoPool' if row['Pool Area'] == 0 else row['Pool QC'], axis=1
)
data['Misc Feature'] = data['Misc Feature'].fillna('None')
data['Alley'] = data['Alley'].fillna('NoAlley')
data['Fence'] = data['Fence'].fillna('NoFence')
data.loc[data['Mas Vnr Area'] == 0, 'Mas Vnr Type'] = 'None'
data = data.dropna(subset=['Mas Vnr Type'])
data.loc[data['Fireplaces'] == 0, 'Fireplace Qu'] = 'None'

data['Lot Frontage'] = data.groupby('Neighborhood')['Lot Frontage'].transform(
    lambda x: x.fillna(x.median()) if x.notna().any() else x
)
data = data.dropna(subset=['Lot Frontage'])

# Garage Data Imputation
garage_columns = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Garage Yr Blt']
for column in garage_columns:
    if column in ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']:
        data.loc[data['Garage Area'] == 0, column] = 'None'
    elif column == 'Garage Yr Blt':
        data.loc[data['Garage Area'] == 0, column] = 0

garage_cars_mode = data.loc[data['Garage Type'] == 'Detchd'].groupby('Neighborhood')['Garage Cars'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 1)
data.loc[(data['Garage Type'] == 'Detchd') & (data['Garage Cars'].isna()), 'Garage Cars'] = data['Neighborhood'].map(garage_cars_mode)

garage_area_median = data.loc[data['Garage Type'] == 'Detchd'].groupby('Neighborhood')['Garage Area'].median()
data.loc[(data['Garage Type'] == 'Detchd') & (data['Garage Area'].isna()), 'Garage Area'] = data['Neighborhood'].map(garage_area_median)

# Basement Data Imputation
bsmt_vars = [
    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1',
    'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF'
]
numeric_bsmt_vars = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
categorical_bsmt_vars = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']

all_bsmt_missing = data[bsmt_vars].isnull().all(axis=1)
data.loc[all_bsmt_missing, numeric_bsmt_vars] = 0
data.loc[all_bsmt_missing, categorical_bsmt_vars] = 'None'



# Handle specific basement-related missing values
data.loc[data['Bsmt Exposure'].isna(), 'Bsmt Exposure'] = 'No'
data.loc[data['BsmtFin Type 2'].isna(), 'BsmtFin Type 2'] = 'Rec'

# Electrical Data Imputation
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

# Add Total SQFT Variable
data['TotalSQFT'] = data['Gr Liv Area'] + data['Total Bsmt SF']

# Add Total Baths Variable
data['TotalBaths'] = data['Full Bath'] + data['Half Bath'] * .5 + data['Bsmt Half Bath'] * .5 + data['Bsmt Full Bath']

# Drop 'Order' and 'PID' because of irrelevance
data.drop(['Order', 'PID'], axis = 1, inplace = True)

# Final Cleanup
data = data.dropna()

count = (data.columns == 'MS SubClass').sum()
print(count)
print(data.dtypes)
data.to_csv('cleaned_data.csv', index=False)

