#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:05:08 2019

@author: jithinj
"""
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot  as plt
data = pd.read_csv("/Users/jithinj/Desktop/data science/course content/dataset/heart.csv")

data.columns
data.dtypes
data.shape


#simple univariant analysis of all columns
for i in data.columns:
    print("**********column name:"+i+"***********")
    print("unique values:")
    print(data[i].unique())
    print("first five values:")
    print(data[i].head())
    print("last five values:") 
    print(data[i].tail())
    plt.figure()
    if len(data[i].unique()) > 5:
        data[i]  = data[i].astype("float64")
        print("Description:") 
        print(data[i].describe())
        sns.distplot(data[i])
        plt.figure()
        sns.boxplot(data[i])
        #plt.figure()
        #if i != "target":
        #    sns.regplot(data[i],data["target"])
        
        
    else:
        data[i]  = data[i].astype("category")
        print("count of each values:")
        print(data[i].value_counts())
        sns.countplot(data[i])

data.dtypes
data.isnull().any(axis=0)

#-----one by one univariant analysis----#

#ob - nill
sns.distplot(data["age"])
sns.boxplot(data["age"])

#ob - almost same
data["sex"].unique()
data["sex"].value_counts()
sns.countplot(data["sex"])

#ob - nill
data["cp"].unique()
data["cp"].value_counts()
sns.countplot(data["cp"])

sns.distplot(data["trestbps"])
sns.boxplot(data["trestbps"])
sns.regplot(data["trestbps"])
#sknewness - outliers

sns.distplot(data["chol"])
sns.boxplot(data["chol"])
#sknewness - outliers

#ob - out of two values one value very less
data["fbs"].unique()
data["fbs"].value_counts()
sns.countplot(data["fbs"])

#ob - out of three value one value is very less
data["restecg"].unique()
data["restecg"].value_counts()
sns.countplot(data["restecg"])

sns.distplot(data["thalach"])
sns.boxplot(data["thalach"])
#skewness remove outliers

data["exang"].unique()
data["exang"].value_counts()
sns.countplot(data["exang"])
#out of two values one is very less

sns.distplot(data["oldpeak"])
sns.boxplot(data["oldpeak"])
#skewness remove outliers

data["slope"].unique()
data["slope"].value_counts()
sns.countplot(data["slope"])
#out of three values one value is very less

data["ca"].unique()
data["ca"].value_counts()
sns.countplot(data["ca"])
#of 5 one is very high

data["thal"].unique()
data["thal"].value_counts()
sns.countplot(data["thal"])
#of four 2 very less

#taget variable - categorical 
data["target"].unique()
data["target"].value_counts()
sns.countplot(data["target"])


#********bivariant analysis********
data.dtypes
#bivariant analysis of all categorical columns with target
from scipy.stats import chi2_contingency
for col in data.columns:
    if col != "target":
        if str(data[col].dtypes) == "category": 
            print("*******"+col+"*****")
            table = pd.crosstab(data["target"],data[col])
            print(chi2_contingency(table))


#bivariant analysis of all continuous columns with target
from scipy.stats import ttest_ind
for col in data.columns:
    if col != "target":
        if str(data[col].dtypes) == "float64": 
            print("*******"+col+"*****")
            target1 = data.loc[data["target"]==1,col]
            target2 = data.loc[data["target"]==0,col]
            print(ttest_ind(target1,target2))
            #sns.regplot(data[col],data["target"])
            #plt.figure()
data.drop(["chol"],inplace=True,axis=1)
data.columns            
data.dtypes

x = data[["age","cp","trestbps","restecg","thalach","exang","oldpeak","slope","ca","thal","sex"]]
y = data["target"]

#---------------Algorithms-------------#


# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# coefficients of logisti regression 
print(classifier.coef_)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n%s" % cm)
accuracy=(sum(cm.diagonal())/cm.sum())*100
print("accuracy is :" ,accuracy)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)*100
print("Accuracy:%s" % acc)



#100% accuracy is over fit so restrict it
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=7,criterion="gini")#"Entropy"
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)

tree.feature_importances_
tree.max_depth

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)*100
print("Accuracy:%s" % acc)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
# using entropy 
ent_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0,oob_score=True)
ent_classifier.fit(X_train, y_train)
#using Gini 
gini_classifier = RandomForestClassifier(n_estimators =10, criterion = 'gini', random_state = 0,oob_score=True)
gini_classifier.fit(X_train, y_train)
#feature importance 
# entropy  
print(ent_classifier.feature_importances_)
print(ent_classifier.oob_score_)
# Gini  
print(gini_classifier.feature_importances_)
print(gini_classifier.oob_score_)

# Predicting the Test set results
# entropy 
y_pred_ent = ent_classifier.predict(X_test)

# gini
y_pred_gini = gini_classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#entrpoy 
acc = accuracy_score(y_test, y_pred_ent)*100
print("Accuracy:%s" % acc)

# Gini 
acc = accuracy_score(y_test, y_pred_gini)*100
print("Accuracy:%s" % acc)

# Fitting Ada boost to the Training set
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

dt = DecisionTreeClassifier() 
lt = LogisticRegression()

classifier = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
#classifier = AdaBoostClassifier(n_estimators=100, learning_rate=1)
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)*100
print("Accuracy:%s" % acc)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)*100
print("Accuracy:%s" % acc)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)*100
print("Accuracy:%s" % acc)



