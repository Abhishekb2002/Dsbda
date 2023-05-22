import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read dataset
df = pd.read_csv('test.csv')
print(df.info)
print(df.head())
print(df.tail())
print(df.shape)
print(df.size)
print(df.dtypes)
print(df.columns)
print(df.sample(5))

print(df.describe())
print(df.isna().sum())

df['Age'].fillna(df['Age'].median(), inplace = True)
print(df.isnull().sum())

fig, axes = plt.subplots(1,2)
fig.suptitle('Boxplot of 1-variable(Age & Fare)')
sns.boxplot(data = df, x = 'Age', ax=axes[0])
sns.boxplot(data = df, x = 'Fare', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,2)
fig.suptitle('Boxplot of 2-variable')
sns.boxplot(data = df, x = 'Survived',y = 'Age', hue = 'Survived', ax=axes[0,0])
sns.boxplot(data = df, x = 'Survived',y = 'Fare', hue = 'Survived', ax=axes[0,1])
sns.boxplot(data = df, x = 'Sex',y = 'Age', hue = 'Sex', ax=axes[1,0])
sns.boxplot(data = df, x = 'Sex',y = 'Fare', hue = 'Sex', ax=axes[1,1])
plt.show()

fig, axes = plt.subplots(1,2)
fig.suptitle('Boxplot of 3-variable(Age & Fare)')
sns.boxplot(data = df, x = 'Sex', y = 'Age', hue = 'Survived', ax=axes[0])
sns.boxplot(data = df, x = 'Sex', y = 'Fare', hue = 'Survived', ax=axes[1])
plt.show()