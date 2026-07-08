import pandas as pd

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Clean Dataset
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)


# Survival Rate by Age Group

df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Adult", "Middle Age", "Senior"]
)

print("Survival Rate by Age Group")

print(df.groupby("Age Group")["Survived"].mean() * 100)


# Survival Rate by Embarkation Port

print("\nSurvival Rate by Embarkation Port")

print(df.groupby("Embarked")["Survived"].mean() * 100)


# Survival Rate by Family Size

df["Family Size"] = df["SibSp"] + df["Parch"]

print("\nSurvival Rate by Family Size")

print(df.groupby("Family Size")["Survived"].mean() * 100)