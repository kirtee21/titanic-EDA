import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Clean Dataset
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)

# Create Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"]


# Age Distribution (Histogram)

plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# Heatmap of Correlations

plt.figure(figsize=(8,6))

correlation = df.select_dtypes(include="number").corr()

sns.heatmap(correlation, annot=True, cmap="Blues")

plt.title("Heatmap of Correlations")
plt.show()


# Survival by Family Size (Bar Plot)

plt.figure(figsize=(6,4))

sns.barplot(x="FamilySize", y="Survived", data=df)

plt.title("Survival by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")

plt.show()