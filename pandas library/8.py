import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(style="darkgrid")
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('test.csv')
print("Information of dataset:\n", df.info)
print("Shape of dataset(row * column):", df.shape)
print("Total elements in dataset:", df.size)
print("Datatype of attributes (columns):\n ",df.dtypes)
print("First 5 rows:\n", df.head())
print("First 5 rows:\n", df.tail().T)
print("First 5 rows:\n", df.sample(5).T)

print(df.describe())
print(df.isna().sum())

df['Age'].fillna(df['Age'].median(), inplace = True)
print(df.isnull().sum())

fig, axes = plt.subplots(1,2)
fig.suptitle('Histogram of 1-variable(Age & Fare)')
sns.histplot(data = df, x = 'Age', ax=axes[0])
sns.histplot(data = df, x = 'Fare', ax=axes[1])
plt.show()

fig, axes = plt.subplots(2,2)
fig.suptitle('Histogram of 2-variable')
sns.histplot(data = df, x = 'Age', hue = 'Survived', multiple = 'dodge',ax=axes[0,0])
sns.histplot(data = df, x = 'Fare', hue = 'Survived', multiple = 'dodge', ax=axes[0,1])
sns.histplot(data = df, x = 'Age',hue = 'Sex', multiple = 'dodge', ax=axes[1,0])
sns.histplot(data = df, x = 'Fare', hue = 'Sex', multiple = 'dodge', ax=axes[1,1])
plt.show()