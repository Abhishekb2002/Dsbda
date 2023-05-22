def RemoveOutlier(df,var):
	Q1=df[var].quantile(0.25) 
	Q3=df[var].quantile(0.75)
	IQR = Q3-Q1 
	high, low=Q3+1.5*IQR, Q1-1.5*IQR
	df=df[((df[var] >= low) & (df[var] <= high))]
	return df

def DisplayOutlier(df, msg):
	fig, axes=plt.subplots(1,2)
	fig.suptitle(msg)
	sns.boxplot(data= df, x='Age', ax=axes[0])
	sns.boxplot(data = df, x='EstimatedSalary', ax=axes[1])
	fig.tight_layout()
	plt.show()

#import Libraries.
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Read Dataset
df=pd.read_csv('Social_Network_Ads.csv')
print('Social Network Ads dataset is successfully loaded.....')

#Display information of dataset
print( 'Information of Dataset:\n', df.info)
print ('Shape of Dataset (row x column):', df.shape)
print('Columns Name:', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes(columns):', df.dtypes)
print('First 5 rows: \n', df.head().T)
print ('Last 5 rows:\n',df.tail().T)
print(' Any 5 rows: \n',df.sample(5).T)

#Find missing values
print('Missing values') 
print(df.isnull().sum())

#Find correlation matrix.
print('Finding correlation matrix using heatmap: ')
sns.heatmap(df.corr(),annot=True) 
plt.show()

#Finding and removing outliers
print('Finding and removing outliers: ')
DisplayOutlier(df, 'Before removing Outliers') 
print( 'Identifying overall outliers in Column Name variables.....')
df=RemoveOutlier(df,'Age')
df=RemoveOutlier(df,'EstimatedSalary') 
DisplayOutlier(df, 'After removing Outliers')

#Split the data into inputs and outputs 
x = df[['Age','EstimatedSalary']] #input data.
y= df['Purchased']  #output data

#Training and testing data:
from sklearn.model_selection import train_test_split

#Assign test data size 20%
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.20, random_state=0)

#Normalization of input data
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test=sc_X.fit_transform(x_test)

#Apply logistic regression model on training data 
from sklearn.linear_model import LogisticRegression 
model=LogisticRegression(random_state = 0, solver='lbfgs') 
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Display classification report 
from sklearn.metrics import classification_report 
print(classification_report(y_test, y_pred))

#Display confusion matrix
from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test, y_pred) 
print('confusion matrix\n',cm) 
fig, axes = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=.3, cmap="Blues") 
plt.show()