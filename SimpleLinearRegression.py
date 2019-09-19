# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:00:53 2019

@author: Venkat Reddy
"""

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')

#Divide the dataset into x and y
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splitting the data based on training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#implement our classifier based on Simple Linear Regression
from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(x_train,y_train)

y_predict=simplelinearRegression.predict(x_test)

#implement the graph
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,simplelinearRegression.predict(x_train))
plt.show()

