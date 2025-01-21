"""
Langvid Phalra

This python file's objective is to determine the growth rate of Ames Properties from 2006 to 2024
- Important note: ZHVI stands for Zillow Home Value Index
"""

# Relevant Libraries
import pandas as pd
import numpy as np
from sklearn.utils.fixes import percentile

# Load Data
data = pd.read_csv('AmesHousing.csv')
city_zhvi_data = pd.read_csv('city_zhvi.csv')

# Setting view options to max
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

"""
Step 1: Calculate median value of Sale Price for Year 2006 - 2009 in Ames
"""

# Removing outliers
percentile_35 = data['SalePrice'].quantile(.35)
percentile_65 = data['SalePrice'].quantile(.65)
ames_data = data[
    (data['SalePrice'] >= percentile_35) & (data['SalePrice'] <= percentile_65)
]

# Filter by year
ames_data = ames_data[
    (ames_data['Yr Sold'] >= 2006) & (ames_data['Yr Sold'] <= 2010)
]

# Calculating median SalePrice
ames_median_price = (
    ames_data.groupby('Yr Sold')['SalePrice'].median().reset_index()
)

print("Median Sale Prices (2006–2009):")
print(ames_median_price)

"""
Step 2: Calculate the annual median SalePrice from 2006 - 2024
"""

# Filter for Ames in ZHVI dataset
ames_zhvi = city_zhvi_data[
    (city_zhvi_data['RegionName'] == 'Ames') & (city_zhvi_data['State'] == 'IA')
]

# Transpose ZHVI columns with date format
zhvi_columns = [col for col in ames_zhvi.columns if '-' in col]
zhvi_transposed = ames_zhvi[zhvi_columns].T
zhvi_transposed.columns = ['ZHVI']
zhvi_transposed = zhvi_transposed.reset_index()
zhvi_transposed.rename(columns={'index': 'Year-Month'}, inplace=True)

# Extract year and calculate annual median
zhvi_transposed['Year'] = zhvi_transposed['Year-Month'].str.split('-').str[0].astype(int)
zhvi_annual_median = zhvi_transposed.groupby('Year')['ZHVI'].median().reset_index()

# Drop rows for years < 2006
zhvi_annual_median = zhvi_annual_median[zhvi_annual_median['Year'] >= 2006]
print("ZHVI Annual Median (2006 onwards):")
print(zhvi_annual_median)

"""
Step 3: Append Ames median prices to ZHVI annual median
"""

# Create a dictionary from the Ames median price DataFrame
ames_median_dict = dict(zip(ames_median_price['Yr Sold'], ames_median_price['SalePrice']))

# Fill missing ZHVI values with Ames median prices
zhvi_annual_median['ZHVI'] = zhvi_annual_median['ZHVI'].fillna(
    zhvi_annual_median['Year'].map(ames_median_dict)
)

print("ZHVI Annual Median with Ames Median Prices:")
print(zhvi_annual_median)

"""
Step 4: Calculate the growth rate from 2006 to 2024
"""

# Ensure 2006 and 2024 exist before calculation
if 2006 in zhvi_annual_median['Year'].values and 2024 in zhvi_annual_median['Year'].values:
    zhvi_2006 = zhvi_annual_median.loc[zhvi_annual_median['Year'] == 2006, 'ZHVI'].values[0]
    zhvi_2024 = zhvi_annual_median.loc[zhvi_annual_median['Year'] == 2024, 'ZHVI'].values[0]

    # Calculate growth rate
    t = 2024 - 2006
    growth_rate = (zhvi_2024 / zhvi_2006) ** (1 / t) - 1
    print(f"Calculated Annual Growth Rate (2006–2024): {growth_rate:.4%}")
else:
    print("Error: Data for 2006 or 2024 is missing.")
