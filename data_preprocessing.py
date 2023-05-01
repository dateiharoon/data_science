# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:54:28 2023

@author: Haroon
"""

#import packages 
import pandas as pd
import matplotlib.pyplot as pit
import numpy as np 
print(np)

#convert excel file to dataframe
df = pd.read_csv('C:/Users/Haroon/Documents/data_science/data_preprocessing/data.csv')

print(df.to_string()) 

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
print(X)
print(y)


from sklearn.impute import SimpleImputer
imputer  = SimpleImputer(missing_values = np.nan, strategy = 'mean' )
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)




from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:,3:] = sc.transform(X_test[:,3:])

print(X_train)
print(X_test)
