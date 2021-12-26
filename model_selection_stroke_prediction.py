#!/usr/bin/env python
# coding: utf-8

# In[1]:

import  preprocessing
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,OneHotEncoder,StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, make_scorer
from sklearn.svm import SVC
import joblib
import pickle


# In[2]:
preprocess

# reading data 

data=pd.read_csv("healthcare-dataset-stroke-data.csv")


# In[3]:


#droped unwanted columns that cant be helfull for predictions 
data.drop(columns='id',axis=1,inplace=True)


# In[4]:


#cheaking all the numerical columns for pipeline creation
cat_col=data.select_dtypes('object').columns
cat_col=np.append(cat_col,['gender', 'ever_married', 'work_type', 'Residence_type',
       'smoking_status'])
cat_col


# In[5]:


# cheaking all the chategorical columns for pipeline 
num_col=data.select_dtypes(exclude='object').columns
num_col


# In[6]:


# bmi feture has missing value that is not good for model performance so we droped it 
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imp_mean.fit(data[['bmi']])
data[['bmi']]=imputer.transform(data[['bmi']])


# In[7]:


# performing train test split on given data 
x_train,x_test,y_train,y_test=train_test_split(data.drop('stroke',axis=1),data.stroke,test_size=0.2,random_state=0)


# In[8]:


#columns trabsformers for all the conversion needed for model in pipeline 

ct=ColumnTransformer([
    ('ohe',OneHotEncoder(sparse=False,handle_unknown='ignore'),cat_col),
    ('ss',StandardScaler(),['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']),

],remainder='passthrough')


# In[9]:


# for hyperparameter tuning we selected some pameter for defferent model 
param_1={
    'model__n_estimators':[10,20],
    'model__max_depth':[2,3]
}


# In[10]:


param_2 = {'model__C':[1,2,3],
             'model__cache_size':[200,300,400],
             }


# In[11]:


param_3={'model' : [LogisticRegression()],
     'model__penalty' : ['l1', 'l2'],
    'model__C' : np.logspace(-4, 4, 20),
    'model__solver' : ['liblinear']}


# In[12]:


#creating pipeline for all three selected model , we have cheaked with only three model in this project 
# in future we ill test with more model 


# In[13]:


pipe1=Pipeline([
    ('ct',ct),
    ('model',RandomForestClassifier(random_state=0,n_jobs=-1))
])


# In[14]:


pipe2=Pipeline([
    ('ct',ct),
    ('model',SVC())
])


# In[15]:


pipe3=Pipeline([
    ('ct',ct),
    ('model',LogisticRegression())
])


# In[16]:


models=[pipe1,pipe2,pipe3]


# In[17]:


# model training and accuracy cheaking 
# hyper-parameter tuning 

    
for i in models:
    if i==pipe1:
        param_grid=param_1
        grid = GridSearchCV(i, param_grid=param_grid, n_jobs=-1)
        grid.fit(x_train,y_train)
        y_prob = grid.best_estimator_.predict_proba(x_test)[:,1]
        roc=roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        print("roc is ",roc)
        print("pr is",pr)
        print(grid.best_score_)
        print(grid.best_estimator_)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    elif i==pipe2:
        param_grid=param_2
        grid = GridSearchCV(i, param_grid=param_grid, n_jobs=-1)
        grid.fit(x_train,y_train)
        y_prob = grid.best_estimator_.predict(x_test)
        roc=roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        print("roc is ",roc)
        print("pr is",pr)
        print(grid.best_score_)
        print(grid.best_estimator_)
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    else :
        param_grid=param_3
        grid = GridSearchCV(i, param_grid=param_grid, n_jobs=-1)
        grid.fit(x_train,y_train)
        y_prob = grid.best_estimator_.predict_proba(x_test)[:,1]
        roc=roc_auc_score(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        print("roc is ",roc)
        print("pr is",pr)
        print(grid.best_score_)
        print(grid.best_estimator_)
    
      
        
        
        


# In[18]:


#model saving which was best among all three model 


# In[19]:


path='new_model'+ '.pkl'


# In[20]:


joblib.dump(grid.best_estimator_,path) 
 


# In[ ]:





# In[ ]:




