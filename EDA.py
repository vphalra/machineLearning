"""
This python file's objective is to use the cleaned Ames data to conduct EDA
"""
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataCleaning import data

data = data
pd.set_option('display.max_rows', 500)
print(data.isna().sum())

# Set display max options to None
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(data.dtypes)

# Scatterplot of Total SQFT and Sale Price
plt.figure(figsize = (10, 6))
plt.scatter(data['TotalSQFT'], data['AdjustedSalePrice'], alpha = .5)
plt.title('Scatter Plot of Total SQFT vs Sale Price', fontsize = 14)
plt.xlabel('Total SQFT', fontsize = 12)
plt.ylabel('Sale Price ($)', fontsize = 12)
plt.grid(True, linestyle = '--', alpha = .6)
plt.show()

# Scatterplot of Gr Liv Area and SalePrice
plt.figure(figsize = (10, 6))
plt.scatter(data['Gr Liv Area'], data['AdjustedSalePrice'], alpha = .5)
plt.title('Scatter Plot of Ground Living Area vs Sale Price', fontsize = 14)
plt.xlabel('Ground Living Area (sq ft)', fontsize = 12)
plt.ylabel('Sale Price ($)', fontsize = 12)
plt.grid(True, linestyle = '--', alpha = .6)
plt.show()

# Histogram of SalePrice
data['AdjustedSalePrice'].hist(bins = 30, edgecolor = 'black')
plt.title('Histogram of Sale Price', fontsize = 14)
plt.xlabel('Sale Price', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show()

# Bar chart of Neighborhood
plt.figure(figsize = (14, 6))
data['Neighborhood'].value_counts().plot(kind='bar')
plt.title('Bar Chart of Neighborhood')
plt.xticks(rotation = 45, fontsize = 10, ha = 'right')
plt.xlabel('Neighborhood', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.tight_layout()
plt.show()

# Scatter plot of SalePrice by Neighborhood
plt.figure(figsize = (14, 7))
plt.scatter(data['Neighborhood'], data['AdjustedSalePrice'], alpha = .5)
plt.xticks(rotation = 45, fontsize = 10, ha = 'right')
plt.title('Scatter Plot Sale Price by Neighborhood', fontsize = 14)
plt.xlabel('Neighborhood', fontsize = 12)
plt.ylabel('Sale Price ($)', fontsize = 12)
plt.tight_layout()
plt.show()

# Scatter plot of SalePrice by SubClass
plt.figure(figsize = (14, 7))
plt.scatter(data['MS SubClass'], data['AdjustedSalePrice'], alpha = .5)
plt.xticks(rotation = 45, fontsize = 10, ha = 'right')
plt.title('Scatter Plot Sale Price by MS SubClass', fontsize = 14)
plt.xlabel('MS SubClass', fontsize = 12)
plt.ylabel('Sale Price ($)', fontsize = 12)
plt.tight_layout()
plt.show()

# Bar chart of Sale Condition
plt.figure(figsize = (14, 6))
data['Sale Condition'].value_counts().plot(kind='bar')
plt.title('Bar Chart of Sale Condition')
plt.xticks(rotation = 45, fontsize = 10, ha = 'right')
plt.xlabel('Sale Condition', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.tight_layout()
plt.show()

"""
An overwhelming majority of the sales recorded were of the "Normal" Condition. So I dropped the rest. 
"""

# Box Plot of Foundation and SalePrice
plt.figure(figsize = (14, 6))
sns.boxplot(x = data['Foundation'], y = data['AdjustedSalePrice'])
plt.title('Box Plot of Foundation and SalePrice')
plt.xlabel('Foundation')
plt.ylabel('Sale Price ($)', fontsize = 12)
plt.xticks(rotation = 45, fontsize = 10, ha = 'right')
plt.tight_layout()
plt.show()

"""
Poured Concrete Foundation consist of the most expensive property sold. Also, with the highest median value. 
"""

# Scatter Plot of Roof Matl vs. SalePrice
plt.figure(figsize=(12, 6))
sns.stripplot(x='Roof Matl', y='AdjustedSalePrice', data = data, jitter = True, alpha=0.7)
plt.title('Scatter Plot of Roof Material vs Sale Price', fontsize=14)
plt.xlabel('Roof Material', fontsize=12)
plt.ylabel('Sale Price ($)', fontsize=12)
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

"""
A majority of properties uses standard composite for their roof material, however, wood shingles had 
the highest priced property. 
"""

# Scatterplot of Land Contour and Adjusted Sale Price
plt.figure(figsize = (12, 6))
sns.scatterplot(x = 'Land Contour', y = 'AdjustedSalePrice', data = data)
plt.title('Scatter Plot of Land Contour and Sale Price')
plt.xlabel('Land Contour', fontsize = 12)
plt.ylabel('Sale Price ($)', fontsize = 12)
plt.show()

"""
Banked contour (steep rise from street to building) has the smallest variance if SalePrice. 
There is a noticeably tight range where price is congested between; 
there are no outliers breaching the threshold of approximately $560k.
"""

import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data: Replace this with your actual DataFrame column
# Example: sale_prices = data['SalePrice']
import numpy as np
np.random.seed(42)
sale_prices = np.random.normal(loc=200000, scale=50000, size=1000)  # Example data

# Plot the distribution of Sale Price
plt.figure(figsize=(10, 6))
sns.histplot(sale_prices, kde=True, bins=30, color="skyblue", edgecolor="black")

# Adding titles and labels
plt.title("Distribution of Sale Price", fontsize=16)
plt.xlabel("Sale Price (USD)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
