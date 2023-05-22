#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read Dataset
df = pd.read_csv('student_data.csv')
print('Dataset loaded into data frame successfully....')

#Display all information
print('**********Information of Dataset********** \n', df.info)
print('**********Shape of Dataset (rows x columns)**********\n',df.shape)
print('**********Columns name**********\n',df.columns)
print('**********Total element in Dataset**********\n',df.size)
print('**********Datatypes of attributes**********\n',df.dtypes)
print('**********First 5 rows**********\n',df.head().T)
print('**********Last 5 rows**********\n',df.tail().T)
print('**********Any 5 rows**********\n',df.sample(5).T)
print('**********Statistical information of numerical columns**********\n',df.describe())
print('**********Total number of null values in dataset**********\n',df.isna().sum())

def RemoveOutlires(df, var):
	Q1 = df[var].quantile(0.25)
	Q3 = df[var].quantile(0.75)
	IQR = Q3 - Q1
	high, low = Q3+1.5*IQR, Q1-1.5*IQR

	df = df[((df[var] >= low) & (df[var] <= high))]
	print('Outlires removed in', var)
	return df

def DisplayOutlires(df, message):
	fig, axes = plt.subplots(2,2)
	fig.suptitle(message)
	sns.boxplot(data = df, x = 'raisedhands', ax = axes[0,0])
	sns.boxplot(data = df, x = 'VisITedResources', ax = axes[0,1])
	sns.boxplot(data = df, x = 'AnnouncementsView', ax = axes[1,0])
	sns.boxplot(data = df, x = 'Discussion', ax = axes[1,1])
	plt.show()

#Handling outliers
DisplayOutlires(df, 'Before removing outliers')
df = RemoveOutlires(df, 'raisedhands')
df = RemoveOutlires(df, 'VisITedResources')
df = RemoveOutlires(df, 'AnnouncementsView')
df = RemoveOutlires(df, 'Discussion')
DisplayOutlires(df, 'After removing outliers')

#Lable encoding
df['gender'] = df['gender'].astype('category')
print('Check datatypes of gender : ', df.dtypes['gender'])
df['gender'] = df['gender'].cat.codes
print('Check datatypes after lable encoding : ', df.dtypes['gender'])
print('gender values : ', df['gender'].unique())


sns.boxplot(data = df, x = 'gender', y = 'raisedhands', hue = 'gender')
plt.title('Boxplot with two variables')
plt.show()

sns.boxplot(data = df, x = 'NationalITy', y = 'Discussion', hue = 'gender')
plt.title("Boxplot with three variables")
plt.show()

sns.scatterplot(data = df, x = 'raisedhands', y = 'VisITedResources')
plt.title('Scatter plot with two variables')
plt.show()
