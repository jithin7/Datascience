#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:04:53 2019

@author: jithinj

"""
# =============================================================================
# AT - air temp
# v - vaccum
# AP - air pressure 
# RH - room humidity
# PE - power generated in energy
#Going to predict Y [PE] using X [AT,V,AP,RH]
# =============================================================================

import pandas as pd
data = pd.read_csv("/Users/jithinj/Desktop/data science/course content/dataset/Folds5x2_pp.csv")
#view columns in the dataset
data.columns
#view the type of variables for the columns
data.dtypes
#first five - to verify the dtype
data.head()
#last five - to verify the dtype
data.tail()
#describe the data - descriptive statistics
desc = data.describe()
print(desc)
#gaining insights from data - inference 
import seaborn as sns
sns.distplot(data["AT"])
sns.boxplot(data["AT"])
sns.distplot(data["V"])
sns.boxplot(data["V"])
sns.distplot(data["AP"])
sns.boxplot(data["AP"])
sns.distplot(data["RH"])
sns.boxplot(data["RH"])
sns.distplot(data["PE"])
sns.boxplot(data["PE"])
#finding correlation between the columns
correlation = data.corr()#observed that correlation between PE and RH is close to zero
print(correlation)
type(correlation)
#Checking the scatter plot if the correlation is affected by outliers which affect the mean
sns.regplot(data["RH"],data["PE"])
#drop the RH column - not used to predict the Y
data.drop(["RH"],inplace=True,axis=1)#inplace-to change in memory,axis - row[0] or column[1]
#check all the scatter plot
sns.regplot(data["AT"],data["PE"])
sns.regplot(data["V"],data["PE"])
sns.regplot(data["AP"],data["PE"])
#checking for triangular effect and droping V column - correlation between AT and V is high
#Also correlation of AT is high than V when comparing with PE
#Hence droping V
data.drop(["V"],inplace=True,axis=1)

#y - prediction variable ; x - features
y =  data["PE"]
x = data[["AP","AT"]]

#sklearn has many inbuilt ML algo which can be used 
#importing linear regression from sklearn
from sklearn.linear_model import LinearRegression
#creating a object 
model = LinearRegression()
#passing features and prediction variable to the model
#fit() does all the calculations[steps for linear regression]
model.fit(x,y)
#
print(model.coef_)
print(model.intercept_)
pred = model.predict(x)

from sklearn.metrics  import r2_score , mean_squared_error
print(r2_score(y,pred))
print(mean_squared_error(y,pred))   
                      

pred_rand = model.predict([])       



