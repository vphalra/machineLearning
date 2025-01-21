import pandas as pd
from pandas import read_csv

# Step 0: Display settings and load data
pd.set_option('display.max_rows', 500)
path = 'cleaned_data.csv'
data = read_csv(path)

# Replace NaN with 'None'
data.fillna('None', inplace=True)


# Encoding MS Subclass separately because of int values as a nominal category
data['MS SubClass'] = data['MS SubClass'].astype('category')
ms_subclass_dummies = pd.get_dummies(data['MS SubClass'], prefix='MS_SubClass', drop_first=True)
data = pd.concat([data, ms_subclass_dummies], axis=1)
data.drop(columns=['MS SubClass'], inplace=True)


# Define variable categories
nominal_vars = [
    'MS Zoning', 'Street', 'Alley', 'Land Contour', 'Lot Config',
    'Neighborhood', 'Condition 1', 'Condition 2', 'Bldg Type', 'House Style',
    'Roof Style', 'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
    'Foundation', 'Heating', 'Central Air', 'Garage Type', 'Misc Feature', 'Sale Type',
    'Sale Condition',
]
ordinal_vars = {
    'Lot Shape': {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1},
    'Utilities': {'AllPub': 2, 'NoSewr': 1},
    'Land Slope': {'Gtl': 3, 'Mod': 2, 'Sev': 1},
    'Overall Qual': {i: i for i in range(1, 11)},  # Assuming 1 to 10
    'Overall Cond': {i: i for i in range(1, 11)},  # Assuming 1 to 9
    'Exter Cond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Exter Qual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2},
    'Bsmt Qual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Bsmt Cond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Bsmt Exposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1},
    'BsmtFin Type 1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1},
    'BsmtFin Type 2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1},
    'Heating QC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Electrical': {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2},
    'Kitchen Qual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
    'Functional': {'Typ': 8, 'Mod': 7, 'Min1': 6, 'Min2': 5, 'Maj1': 4, 'Maj2': 3},
    'Fireplace Qu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
    'Garage Finish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0},
    'Garage Qual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
    'Garage Cond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
    'Paved Drive': {'Y': 3, 'P': 2, 'N': 1},
    'Pool QC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'NoPool': 0},
    'Fence': {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NoFence': 0}
}

# Step 1: Encode nominal variables
# Verify which nominal variables exist in the data
present_nominal_vars = [col for col in nominal_vars if col in data.columns]
missing_nominal_vars = [col for col in nominal_vars if col not in data.columns]

# One-hot encode nominal variables
nominal_dummies = pd.get_dummies(data[present_nominal_vars], prefix=present_nominal_vars, drop_first=True)

# Concatenate dummies and drop original nominal columns
data = pd.concat([data, nominal_dummies], axis=1)
data.drop(columns=present_nominal_vars, inplace=True)

# Step 2: Ordinal encoding
for column, mapping in ordinal_vars.items():
    if column in data.columns:
        data[column] = data[column].map(mapping).fillna(0)  # Map and fill missing with 0
    else:
        print(f"Warning: Column '{column}' not found in dataset. Skipping.")


