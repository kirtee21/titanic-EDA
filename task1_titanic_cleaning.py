import pandas as pd

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Check Missing Values
print(df.isnull().sum())

# Fill missing values in Age with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Drop irrelevant column
df.drop("Cabin", axis=1, inplace=True)

# Check dataset after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)