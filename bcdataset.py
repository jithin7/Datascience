# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:20:51 2019

@author: Kavitha Shiva
"""

import pandas as pd
import seaborn as sns
data=pd.read_excel("K:/jithin/DataScience/Course Materials/bcdataset.xlsx")
print(data.columns)

data.columns = [i.replace(" ","") for i in data.columns]


print(data.dtypes)
print(data.shape)
print(data.head)
print(data.tail)

for col in data.columns:
    print(col)
    print(data[col].unique())


data.drop(["id","Unnamed:11"],inplace=True,axis=1)
print(data.columns)

data.isnull().any(axis=0)
data.loc[data["BareNuclei"]=="?","BareNuclei"].count()
#data["BareNuclei"]=data["BareNuclei"].replace("?",1)
data.loc[data["BareNuclei"]=="?","BareNuclei"] = 1

data["BareNuclei"].value_counts()

data.loc[data["diagnosis"]==2,"diagnosis"] = 0
data.loc[data["diagnosis"]==4,"diagnosis"] = 1


from scipy.stats import chi2_contingency
for col in data.columns:
    if col != "diagnosis":
        print(col)
        table = pd.crosstab(data["diagnosis"],data[col])
        print(chi2_contingency(table))
        
x = data.drop(["diagnosis"],axis=1)
y = data["diagnosis"]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


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
tree.fit(x,y)
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
