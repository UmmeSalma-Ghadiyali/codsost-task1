import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('titanic-dataset.csv')

# Display basic information about the dataset
print("Data Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Data visualization
sns.set(style='whitegrid')

# Countplot for Survived
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Countplot for Sex
sns.countplot(x='Sex', data=df)
plt.title('Gender Count')
plt.show()

# Histogram for Age
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot for Fare
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Pclass')
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
