import numpy as np 
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt

# Read Dataset

df=pd.read_csv('iris.csv')
print('Iris dataset is successfully loaded')

#Display information of dataset

print('Information of Dataset:\n', df.info)

print('Shape of Dataset (row x column):\n' ,df. shape)

print('Columns Name:\n', df.columns)

print('Total elements in dataset:\n', df.size)
print('Datatype of attributes (columns)\n', df.dtypes)


print('First 5 rows: \n', df.head().T)

print('Last 5 rows: \n',df.tail().T)

print( 'Any 5 rows: \n',df.sample(5).T)

# @Find missing values:

print ('Missing values') 
print(df.isnull().sum())

# Histogram of I-variable 
fig, axes= plt.subplots(2,2)

fig.suptitle( 'Histogram 1-variables')

sns.histplot(data=df, x='sepal.length', ax=axes[0,0])

sns.histplot(data=df, x='sepal.width', ax=axes[0,1])
sns.histplot(data= df, x='petal.length', ax=axes[1,0])

sns.histplot(data = df, x='petal.width', ax=axes[1,1])

plt.show()

# Histogram of 2-variables

fig, axes= plt.subplots(2,2)

fig.suptitle('Histogram of 2-variables') 
sns.histplot(data = df, x='sepal.length',hue='variety', multiple= 'dodge', ax=axes[0,0])

sns.histplot(data=df, x='sepal.width',hue='variety', multiple='dodge', ax=axes[0,1])
sns.histplot(data=df, x='petal.length',hue='variety', multiple= 'dodge', ax=axes[1,0])
sns.histplot(data = df, x='petal.width',hue='variety', multiple ='dodge', ax=axes[1,1])
plt.show()



# Boxplot of 1-variable.

fig,axes= plt.subplots(2,2)

fig.suptitle(' Boxplot of 1-variables')

sns.boxplot(data = df, x = 'sepal.length', ax=axes[0,0])

sns.boxplot(data= df, x='sepal.width', ax=axes[0,1]) 
sns.boxplot(data= df, x='petal.length', ax= axes[1,0])
sns.boxplot(data= df, x='petal.width', ax =axes[1,1])
plt.show()



#Boxplot of 2-variables

fig,axes=plt.subplots(2,2) 
fig.suptitle('Boxplot of 2-variables')
sns.boxplot(data = df, x='sepal.length', y='variety', hue='variety', ax=axes[0,0])
sns.boxplot(data=df, x='sepal.width', y='variety', hue='variety',ax=axes[0,1])
sns.boxplot(data = df, x = 'petal.length', y='variety', hue='variety', ax=axes[1,0])
sns.boxplot(data = df, x='petal.width', y='variety', hue='variety',ax=axes[1,1])
plt.show()