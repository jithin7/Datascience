# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:01:03 2019

@author: Kavitha Shiva
"""
import pandas as pd
import seaborn as sns
data=pd.read_csv("K:/jithin/DataScience/Course Materials/Training_by_me_light/4) ML/Lab_class/Logistic-Regression/Social_Network_Ads.csv")
print(data.columns)
print(data.dtypes)
print(data.shape)
print(data.head)#smtyms they may vary so check
print(data.tail)
data.drop(["User ID"],inplace=True,axis=1)
#Dropping User ID
print(data.columns)

desc=data.describe()#finding mean,std.... of all the indiv columns
print(desc)
#sns.distplot(data["Gender"])
#sns.boxplot(data["Gender"])
#not possi for gen as bino
data["Gender"].unique()
data["Gender"].value_counts()


sns.distplot(data["Age"])
sns.boxplot(data["Age"])
sns.distplot(data["EstimatedSalary"])
sns.boxplot(data["EstimatedSalary"])
###########
#not possi for gen as categorical
data["Purchased"].unique()
data["Purchased"].value_counts()
# imbalance class prob as 0-257 & 1-143 which pur is less than non purch
##############

#comparing pur(cat)nd Gen(cat) so start chi sq with freq table  crosstab creates 2way freq table
pur_gen=pd.crosstab(data["Purchased"],data["Gender"])
print(pur_gen)

from scipy.stats import chi2_contingency
print(chi2_contingency(pur_gen))
#chisq,pvalue,deg of freedom,expctd value of array
#Dropping gender based on pvalue as it is 0.45 which is not <0.05
data.drop(["User ID"],inplace=True,axis=1)

####performin t test on conti conti values
###AGE##
#syntax -data.loc[row,col] can be used for slicing eg[0:5,3]selects first 5 rows of 3rd column
Age_1=data.loc[data["Purchased"]==1,"Age"]
Age_0=data.loc[data["Purchased"]!=1,"Age"]
from scipy.stats import ttest_ind
print(ttest_ind(Age_1,Age_0))
#(related as per p val)


###ESTimated salary
EstimatedSalary_1=data.loc[data["Purchased"]==1,"EstimatedSalary"]
EstimatedSalary_0=data.loc[data["Purchased"]!=1,"EstimatedSalary"]
from scipy.stats import ttest_ind
print(ttest_ind(EstimatedSalary_1,EstimatedSalary_0))
#(related as per p val)
sns.regplot(data["Age"],data["Purchased"])
sns.regplot(data["EstimatedSalary"],data["Purchased"])
#both does not fit in linear regr so fit in curve


x = data[["EstimatedSalary","Age"]]
y = data["Purchased"]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
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

import numpy as np
import matplotlib.pyplot as plt
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




