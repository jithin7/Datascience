#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:31:40 2019

@author: jithinj
"""

import pandas as pd
import seaborn as sns
import  matplotlib.pyplot  as plt
data = pd.read_csv("/Users/jithinj/Desktop/data science/course content/dataset/Churn_Modelling.csv")

data.columns
data.dtypes
data.shape

#columns dropped 
data.drop(["RowNumber"],inplace=True,axis=1)
data.drop(["CustomerId"],inplace=True,axis=1)
data.drop(["Surname"],inplace=True,axis=1)

#simple univariant analysis of all the other columns
for i in data.columns:
    print("**********column name:"+i+"***********")
    print("unique values:")
    print(data[i].unique())
    print("first five values:")
    print(data[i].head())
    print("last five values:") 
    print(data[i].tail())
    fig = plt.figure(figsize=(8,3))
    if len(data[i].unique()) > 5:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        data[i]  = data[i].astype("float64")
        print("Description:") 
        print(data[i].describe())
        #ax1.hist(data[i])
        sns.distplot(data[i],ax=ax1)
        #ax2.boxplot(data[i])
        sns.boxplot(data[i],ax=ax2)
        #plt.show()
    else:
        #ax1 = fig.add_subplot(121)
        data[i]  = data[i].astype("category")
        print("count of each values:")
        print(data[i].value_counts())
        sns.countplot(data[i])

data.dtypes
data["Tenure"]  = data["Tenure"].astype("category")
data.dtypes

#********bivariant analysis********
#bivariant analysis of all categorical columns with target
from scipy.stats import chi2_contingency
for col in data.columns:
    if col != "Exited":
        if str(data[col].dtypes) == "category": 
            print("*******"+col+"*****")
            table = pd.crosstab(data["Exited"],data[col])
            print(chi2_contingency(table))


#bivariant analysis of all continuous columns with target
from scipy.stats import ttest_ind
for col in data.columns:
    if col != "Exited":
        if str(data[col].dtypes) == "float64": 
            print("*******"+col+"*****")
            target1 = data.loc[data["Exited"]==1,col]
            target2 = data.loc[data["Exited"]==0,col]
            print(ttest_ind(target1,target2))
            #sns.regplot(data[col],data["target"])
            #plt.figure()

data.columns            
data.drop(["NumOfProducts"],inplace=True,axis=1)
data.columns            
data.dtypes

x = data[["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"]]
y = data["Exited"]