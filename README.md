# Coursera_project

#Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#Load dataset
url = "https://raw.githubusercontent.com/callxpert/datasets/master/Loan-applicant-details.csv"
names = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
dataset = pd.read_csv(url, names=names)

#Convert String data(Labels) to integer
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])

# Splitting Train and test data
array = dataset.values
X = array[:,6:11]
X=X.astype('int')        #To make sklearn aware of 'int' type of X
Y = array[:,12]
Y=Y.astype('int')        #To make sklearn aware of 'int' type of Y

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.48, random_state=8)

#regression model
Log_reg=LogisticRegression()
Log_reg.fit(x_train,y_train)
y_pred=Log_reg.predict(x_test)
print(y_pred)

#confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))

#Accuracy of the Model
print("Accuracy of model: ",accuracy_score(y_test, y_pred))

# Decision Tree Classifier

from sklearn import tree

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=6)

Dec_tree=tree.DecisionTreeClassifier()
Dec_tree.fit(x_train,y_train)
y_pred=Dec_tree.predict(x_test)
print(y_pred)

#confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))

#Accuracy of the Model
print("Accuracy of model: ",accuracy_score(y_pred,y_test))

#Support Vector Machine (SVM)Classifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

model_to_set = OneVsRestClassifier(SVC(kernel="poly"))

param_grid = {
    "estimator__C": [10,20,4,8],
    "estimator__kernel": ["poly","rbf"],
    "estimator__degree":[1, 2, 3, 4],
}

model_tunning = GridSearchCV(model_to_set,param_grid, scoring = 'f1_micro', cv=6,n_jobs=-1) #Tunning parameters


model_tunning.fit(x_train,y_train)
print ("Accuracy: ",model_tunning.best_score_)   #Best score among provided parameters
print ("Parameters for this Accuracy: ",model_tunning.best_params_)

#k-Nearest Neighbour Classifier
from sklearn.neighbors import KNeighborsClassifier 

knn_model = KNeighborsClassifier(n_neighbors=5)   #Efficient neighbors value for highest accuracy
  
knn_model.fit(x_train, y_train) 
y_pred=knn_model.predict(x_test)
print("Accuracy of model: ",accuracy_score(y_test, y_pred))

#confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
