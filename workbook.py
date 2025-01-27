import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.try_ols_anova import anova_str

path = "cleaned_data.csv"
data = read_csv(path)


# Calculate the overall median sale price
overall_median_sale_price = data['SalePrice'].median()
print(f"Overall Median Sale Price: ${overall_median_sale_price:,.2f}")

summary = data['SalePrice'].describe()
print(summary)

pd.options.display.float_format = '{:,.2f}'.format
# Calculate the overall median sale price
overall_median_sale_price = data['AdjustedSalePrice'].median()
print(f"Overall Median Sale Price: ${overall_median_sale_price:,.2f}")

summary = data['AdjustedSalePrice'].describe()
print(summary)

# Summary Stats Visualization for sale price and adjusted sale price
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


# Function to plot bar charts with improved annotation positioning
def plot_summary_statistics(categories, values, title):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=14)
    plt.xlabel('Statistic', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding value annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Adjusting position for annotations
        if i == len(bars) - 1:  # For the Max bar, position annotation in the middle
            plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"${int(height):,}",
                     ha='center', fontsize=10, color='white', weight='bold')
        else:  # Position others slightly above the bar
            plt.text(bar.get_x() + bar.get_width() / 2, height + (height * 0.03), f"${int(height):,}",
                     ha='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()


import seaborn as sns

import matplotlib.pyplot as plt


# Function to plot bar charts with clean annotation placement
def plot_summary_statistics(categories, values, title):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, values, color='skyblue', edgecolor='black')
    plt.title(title, fontsize=14)
    plt.xlabel('Statistic', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adding annotations with consistent padding
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # Center text on the bar
            height + 5000,  # Slightly above the bar
            f"${int(height):,}",  # Format as currency
            ha='center', va='bottom',  # Center horizontally, align text bottom
            fontsize=10, color='black'
        )

    plt.tight_layout()
    plt.show()


# Data for original sale prices
stats = {
    "Min": 35000,
    "25% (Q1)": 130062.5,
    "Median (Q2)": 160000,
    "75% (Q3)": 207500,
    "Max": 755000
}
categories = list(stats.keys())
values = list(stats.values())

# Plot original sale prices
plot_summary_statistics(categories, values, 'Summary Statistics of Original Sale Prices')

# Data for adjusted sale prices
adjustedStats = {
    "Min": 55135.60,
    "25% (Q1)": 197221.90,
    "Median (Q2)": 241135.18,
    "75% (Q3)": 311059.39,
    "Max": 1159701.76
}
adjustedCategories = list(adjustedStats.keys())
adjustedValues = list(adjustedStats.values())

# Plot adjusted sale prices
plot_summary_statistics(adjustedCategories, adjustedValues, 'Summary Statistics of Adjusted Sale Prices')


# Plot the distribution of Sale Price
plt.figure(figsize=(10, 6))
sns.histplot(data['AdjustedSalePrice'], kde=True, bins=30, color="skyblue", edgecolor="black")

# Adding titles and labels
plt.title("Distribution of Sale Price", fontsize=16)
plt.xlabel("Sale Price (in Million $)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# box plot of neighborhood and sale price
plt.figure(figsize=(14, 8))
sns.boxplot(x= 'Neighborhood', y = 'AdjustedSalePrice', data = data, palette = 'coolwarm')
plt.title('Sale Prices by Neighborhood')
plt.xlabel('Neighborhood in Ames')
plt.ylabel('Sale Price (in Million $)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import scipy.stats as stats

# Perform ANOVA
anova_result = stats.f_oneway(*[data[data['Neighborhood'] == n]['AdjustedSalePrice'] for n in data['Neighborhood'].unique()])
print('ANOVA p-value:', anova_result.pvalue)

print(anova_result)

# box plot of overall quality and sale price
plt.figure(figsize=(14,8))
sns.boxplot(x = 'Overall Qual'
            , y = 'AdjustedSalePrice',
            data = data)
plt.title('Sale Prices based on Overall Quality')
plt.xlabel('Overall Quality')
plt.ylabel('Sale Price (in Million $)', fontsize=14)
plt.tight_layout()
plt.show()

from scipy.stats import f_oneway

anova_result = f_oneway(*[data[data['Overall Qual'] == qual]['AdjustedSalePrice'] for qual in data['Overall Qual'].unique()])
print('ANOVA p-value:', anova_result)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Create a 'DecadeBuilt' column
data['DecadeBuilt'] = (data['Year Built'] // 10) * 10

# Box plot for sale prices grouped by decades
plt.figure(figsize=(12, 8))
sns.boxplot(x='DecadeBuilt', y='AdjustedSalePrice', data=data, palette='coolwarm')
plt.title('Sale Prices by Decade Built', fontsize=16)
plt.xlabel('Decade Built', fontsize=14)
plt.ylabel('Sale Price (USD)', fontsize=14)
plt.xticks(rotation=45)
plt.show()


import pandas as pd
import scipy.stats as stats

# Group by decades and calculate sale prices
data['Decade'] = (data['Year Built'] // 10) * 10

# Perform ANOVA test
groups = [group['AdjustedSalePrice'] for _, group in data.groupby('Decade')]
f_stat, p_value = stats.f_oneway(*groups)

print("ANOVA Test:")
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("The year built (decade) is statistically significant in determining sale prices.")
else:
    print("The year built (decade) is not statistically significant in determining sale prices.")


# scatter plot of garage cars and sale price
plt.figure(figsize=(12, 8))
sns.scatterplot(x = 'Garage Cars', y = 'AdjustedSalePrice', data = data, alpha = .6, color = 'blue')
plt.title('Sale Prices by Garage Capacity', fontsize=16)
plt.xlabel('Garage Cars', fontsize=14)
plt.ylabel('Sale Price (in Million $)', fontsize=14)
plt.show()

garage_groups = [group['AdjustedSalePrice'] for _, group in data.groupby('Garage Cars')]

# Perform ANOVA test
f_stat, p_value = stats.f_oneway(*garage_groups)

# Print the results
print("ANOVA Test Results:")
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("The number of garage spaces has a statistically significant effect on sale prices.")
else:
    print("The number of garage spaces does not have a statistically significant effect on sale prices.")

pd.set_option('display.max_rows', None)
print(data.dtypes)

categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
# Add known integer-encoded categorical variables
categorical_columns += ['MS SubClass']  # Add other integer-encoded categorical columns as needed

# Exclude categorical columns to focus only on true numerical variables
numerical_df = data.drop(columns=categorical_columns)

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, dendrogram

# Perform hierarchical clustering on the correlation matrix
linkage_matrix = linkage(correlation_matrix, method='ward')

# Plot a dendrogram
plt.figure(figsize=(14, 9))
dendrogram(linkage_matrix, labels=correlation_matrix.columns, leaf_rotation=90)
plt.title("Dendrogram of Correlated Features", fontsize=16)
plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
categorical_columns += ['MS SubClass']
numerical_df = data.drop(columns=categorical_columns)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_df)
scaled_df = pd.DataFrame(scaled_data, columns = numerical_df.columns)

pca = PCA()
pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_result, columns = [f'PC{i+1}' for i in range(pca_result.shape[1])])

print(correlation_matrix)

# Set a correlation threshold
correlation_threshold = 0.7

# Find variable pairs with high correlation
high_corr_pairs = correlation_matrix.unstack()  # Unstack the matrix to get variable pairs
high_corr_pairs = high_corr_pairs[high_corr_pairs.abs() > correlation_threshold]  # Filter pairs above the threshold
high_corr_pairs = high_corr_pairs[high_corr_pairs != 1]  # Remove self-correlations (correlation = 1)

# Drop duplicate pairs (e.g., (A, B) and (B, A))
high_corr_pairs = high_corr_pairs.sort_values(ascending=False).drop_duplicates()

# Display highly correlated pairs
print(high_corr_pairs)
print(numerical_df.dtypes)

selected_vars = [
    'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces',
    'Garage Yr Blt', 'Garage Cars', 'Garage Area', 'Wood Deck SF',
    'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch',
    'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold', 'SalePrice',
    'Year Remod/Add', 'Mas Vnr Area', 'Lot Frontage', 'Lot Area',
    'Overall Qual', 'Overall Cond'
]
corr = numerical_df[selected_vars].corr()
print(corr)

high_corr_pairs = corr.unstack()  # Unstack the matrix to get variable pairs
high_corr_pairs = high_corr_pairs[high_corr_pairs.abs() > correlation_threshold]  # Filter pairs above the threshold
high_corr_pairs = high_corr_pairs[high_corr_pairs != 1]  # Remove self-correlations (correlation = 1)

# Drop duplicate pairs (e.g., (A, B) and (B, A))
high_corr_pairs = high_corr_pairs.sort_values(ascending=False).drop_duplicates()

# Display highly correlated pairs
print(high_corr_pairs)
print(numerical_df.dtypes)

numerical_df.drop(columns = ['Garage Area', 'Year Built', 'BsmtFin SF 1', 'BsmtFin SF 2',
                             'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF',
                             'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath'])