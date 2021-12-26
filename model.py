
# importing recuire libraries for model

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score,classification_report,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle


#data importing
X=pd.read_csv("x.csv")
y=pd.read_csv("y.csv")


#feture selection and target selection
column=[ 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']


x=X[column]
y=y['stroke']

#data spliting
X_train, X_valid, y_train, y_valid = train_test_split(x, y,test_size=.2, random_state=42)

#shape checking
print(X_train.shape,y_train.shape)
print(X_valid.shape,y_valid.shape)

#model buiding
model=SVC()
model.fit(X_train,y_train)
predicted=model.predict(X_valid)
print(accuracy_score(y_valid,predicted))