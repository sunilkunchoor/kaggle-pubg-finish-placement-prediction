"""
PUBG Finish Placement Prediction

Created on Tue Oct  9 22:27:02 2018

@author: Sunil Kunchoor

In this notebook, I will test different classification menthods on the given data and find the precision.

1. Logical Regression
2. KNN
3. SVM
4. Decision Tree
5. Random Forrest
"""
#########################################################
############  DATA PREPROCESSING    #####################
#########################################################

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing and viewing the dataset
dataset = pd.read_csv("data/train.csv")
dataset.head()


#looking the type and searching for null values
dataset.info()


# Checking the targted classes using Histogram
plt.hist(dataset['Class'], bins = [0,0.5,1], range = (0,1), rwidth = 0.5, align = 'left')
plt.title("Fraud class Histogram")
plt.xticks([0,0.5],["Not Fraud","Fraud"])
plt.xlabel("Is it a fraud?")
plt.ylabel("Count")
plt.show()

# Plotting all the values wrt to time
# for i in xrange(1, 29):
plt.plot(dataset['Time'],dataset['V1'])
plt.show()

# Creating X and y from the data
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


##########################################################
############  LOGISTIC REGRESSION    #####################
##########################################################

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

# Predicting the Test set results
lr_y_pred = lr_classifier.predict(X_test)

# Checking the accuracy using the Area Under the Precision-Recall Curve
from sklearn.metrics import average_precision_score
lr_avg_precision = average_precision_score(y_test, lr_y_pred)



########################################################################
############  K NEAREST NEIGHBOR CLASSIFICATION    #####################
########################################################################

# Fitting KNN classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

# Predicting the Test set results
knn_y_pred = knn_classifier.predict(X_test)

# Checking the accuracy using the Area Under the Precision-Recall Curve
from sklearn.metrics import average_precision_score
knn_avg_precision = average_precision_score(y_test, knn_y_pred)



############################################################################
############  SUPPORT VECTOR MACHINE CLASSIFICATION    #####################
############################################################################

# Fitting SVM classifier to the Training set
from sklearn.svm import SVC
svm_classifier = SVC(kernel="linear", random_state=0)
svm_classifier.fit(X_train, y_train)

# Predicting the Test set results
svm_y_pred = svm_classifier.predict(X_test)

# Checking the accuracy using the Area Under the Precision-Recall Curve
from sklearn.metrics import average_precision_score
svm_avg_precision = average_precision_score(y_test, svm_y_pred)



###################################################################
############  DECISION TREE CLASSIFICATION    #####################
###################################################################

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
dt_classifier.fit(X_train, y_train)

# Predicting the Test set results
dt_y_pred = dt_classifier.predict(X_test)

# Checking the accuracy using the Area Under the Precision-Recall Curve
from sklearn.metrics import average_precision_score
dt_avg_precision = average_precision_score(y_test, dt_y_pred)



###################################################################
############  RANDOM FOREST CLASSIFICATION    #####################
###################################################################

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
rf_classifier.fit(X_train, y_train)

# Predicting the Test set results
rf_y_pred = rf_classifier.predict(X_test)

# Checking the accuracy using the Area Under the Precision-Recall Curve
from sklearn.metrics import average_precision_score
rf_avg_precision = average_precision_score(y_test, rf_y_pred)




