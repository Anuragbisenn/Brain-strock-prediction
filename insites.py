#!/usr/bin/env python
# coding: utf-8

# #### Introduction:
# #### Strokes are the second leading cause of death and the third leading cause of disability globally. Stroke is the sudden death of some brain cells due to lack of oxygen when the blood flow to the brain is lost by blockage or rupture of an artery to the brain
# #### Probilam statement:
# #### This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like Gender, Age, various diseases
# #### Objective:
# #### Constract a preditive model for predicting stroke and to aassess the accuracy of the medels.We will apply and  explore 7 algorithms to see which produces reliable and repeatable results. They  are: Decision Tree,Logistic Regression,Random Forest,SVM,KNN,Naive Bayes,KMeans Clustering.
# #### Data Source:
# ##### A population of 5110 people are involved in this study with 2995 females and 2115 males. The dataset for this study is extracted from Kaggle data respositories.

# In[1]:


import numpy as np
import pandas as pd
from IPython import  get_ipython
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.rcParams['figure.figsize'] = (5,5)
from sklearn.metrics import accuracy_score, f1_score,classification_report,precision_score,recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import pickle


# In[2]:


#get_ipython().system('pip install imblearn')


# In[3]:


from imblearn.over_sampling import SMOTE


# data importing

data = pd.read_csv('healthcare-dataset-stroke-data.csv')


# In[5]:


data


# In[ ]:





# #### EDA

# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.shape


# In[9]:


data['bmi'].value_counts()


# In[10]:


data.describe().T


# In[11]:


data['bmi'].describe()


# In[12]:


data['bmi'].fillna(data['bmi'].mean(),inplace=True)


# In[13]:


data.isnull().sum()


# In[14]:


data.drop('id',axis=1,inplace=True)


# In[15]:


#gender wise comparision of stroke rate
data['gender'].value_counts()


# In[16]:


# Gender feature needs to be converted to  binary variable.Hence, we will impute this single value with mode in this column.
data['gender'] = data['gender'].replace('Other', list(data.gender.mode().values)[0])
data.gender.value_counts()


# In[17]:


#Stroke probabilities gender wise, it shows dataset is imbalanced no much difference between male and female
sns.countplot(data=data,x='gender',hue='stroke')


# In[18]:


data['work_type'].unique()


# #### Categorical Feature Analysis

# In[19]:


df_cat = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status','stroke']

fig, axs = plt.subplots(4, 2, figsize=(14,20))
axs = axs.flatten()

# iterate through each column of df_catd and plot
for i, col_name in enumerate(df_cat):
    sns.countplot(x=col_name, data=data, ax=axs[i], hue =data['stroke'], palette = 'flare')
    plt.title("Bar chart of")
    axs[i].set_xlabel(f"{col_name}", weight = 'bold')
    axs[i].set_ylabel('Count', weight='bold')


# #### Observations from the count plot are listed below:
# #### Hypertension:hypertension have highly risk of having stroke.hypertension is rare in young people and common in aged people.But we only quite little data  on patients having hypertension.
# #### Heart disease:Subjects that previously diagnosed with heart disease have highly risk of having stroke.  
# #### Ever married: People who are married have a higher stroke rate.
# #### Work type:People working in the Private sector have a higher risk of getting a stroke. And people who have never worked have a very less stroke rate.  
# #### Residence type:This attribute is of no use. As we can see there not much difference in both attribute values. Maybe we have to discard it.
# #### smoking status:As per these plots, we can see there is not much difference in the chances of stroke irrespective of smoking status.

# #### Numerical Feature Analysis

# In[20]:


df_num = ['age', 'avg_glucose_level', 'bmi']

fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs = axs.flatten()

# iterate through each column in df_num and plot
for i, col_name in enumerate(df_num):
    sns.boxplot(x="stroke", y=col_name, data=data, ax=axs[i],  palette = 'Set1')
    axs[i].set_xlabel("Stroke", weight = 'bold')
    axs[i].set_ylabel(f"{col_name}", weight='bold')


# #### Age and Stroke:it shows an higher mean agebut no prominant observation of how bmiaffects the chance of having a stroke
# ##### Stroke & avg_glucose level: strokes tends to have an higher average glucose level of more than 100
# #### Bmi and stroke:Bmi does not give much indication on how does it affects the chance of having a stroke

# In[21]:


data.head()


# #### Outlier removel

# In[22]:


data.plot(kind="box")
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# In[23]:


data['avg_glucose_level'].describe()


# In[24]:


data[data['avg_glucose_level']>114.090000]


# In[25]:


data['work_type'].value_counts()


# In[26]:


data.info()


# In[27]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()


# In[28]:


gender = enc.fit_transform(data['gender'])
ever_married = enc.fit_transform(data['ever_married'])
work_type = enc.fit_transform(data['work_type'])
residence_type = enc.fit_transform(data['Residence_type'])
smoking_status = enc.fit_transform(data['smoking_status'])


# In[29]:


data['gender'] = gender
data['ever_married'] = ever_married
data['work_type'] = work_type
data['Residence_type'] = residence_type
data['smoking_status'] = enc.fit_transform(data['smoking_status'])


# In[30]:


data.head()




data.info()




#4.9% of the population in this dataset is diagnosed with stroke
plt.figure(figsize=(4,4))
data['stroke'].value_counts().plot.pie(autopct='%1.1f%%', colors = ['#66b3ff','#99ff99'])
plt.title("Pie Chart of Stroke Status", fontdict={'fontsize': 14})
plt.tight_layout()



plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot=True,fmt = '.2');


# #### We can see variable (attributes) that are showing some effective correlation are:age, hypertension, heart_disease, ever_married, avg_glucose_level.


# #### Spliting for train and test



X=data.drop('stroke',axis=1)
X
X.to_csv('X_ab1.csv',index=True)



classifier = SelectKBest(score_func=f_classif,k=5)
fits = classifier.fit(data.drop('stroke',axis=1),data['stroke'])
x=pd.DataFrame(fits.scores_)
columns = pd.DataFrame(data.drop('stroke',axis=1).columns)
fscores = pd.concat([columns,x],axis=1)
fscores.columns = ['Attribute','Score']
fscores.sort_values(by='Score',ascending=False)


#let’s check our features using SelectKBest and F_Classif.
#Cheching with the features got in heatmap and keep the threshold value as 50
cols=fscores[fscores['Score']>50]['Attribute']
print(cols)

Y=data['stroke']
Y
Y.to_csv('Y_ab1.csv',index=True)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=101)


X_test
Y_test


# #### Normalisation


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)



X_train_std


# #### Training

# ##### Decision Tree


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()



dt.fit(X_train_std,Y_train)

dt.feature_importances_
X_train.columns
Y_pred = dt.predict(X_test_std)
Y_pred
Y_test
X_test


from sklearn.metrics import accuracy_score
ac_dt = accuracy_score(Y_test,Y_pred)
ac_dt


# #### LogisticRegression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_std,Y_train)
Y_pred = lr.predict(X_test_std)
Y_pred
Y_test
ac_lr = accuracy_score(Y_test,Y_pred)
ac_lr


# #### KNN


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_std,Y_train)
y_pred = knn.predict(X_test_std)
y_pred
acc_knn = accuracy_score(Y_test,Y_pred)
acc_knn


# #### Random Forest


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
Y_pred = rf.predict(X_test)
Y_pred
acc_rf = accuracy_score(Y_test,Y_pred)
acc_rf


# #### SVM
from sklearn.svm import SVC
sv = SVC()
sv.fit(X_train_std,Y_train)
Y_pred = sv.predict(X_test)
Y_pred
ac_sv = accuracy_score(Y_test,Y_pred)
ac_sv


#ALL models evoluation


models = dict()
models['Decision Tree'] = DecisionTreeClassifier()
models['Logistic Regression'] = LogisticRegression()
models['Random Forest'] = RandomForestClassifier()
models['Support Vector Machine'] = SVC(kernel = 'sigmoid', gamma='scale')
models['kNN'] = KNeighborsClassifier()
models['Naive Bayes'] = GaussianNB()
models['KMeans'] = KMeans(n_clusters=2, n_init=10, random_state=42)
for model in models:
    
    models[model].fit(X_train, Y_train)
    print(model + " model fitting completed.")



# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 101)


for x in models:

    print('-'*20+x+'-'*20)
    model = models[x]
    Y_pred = model.predict(X_test)
    arg_test = {'y_true':Y_test, 'y_pred':Y_pred}
    print(confusion_matrix(**arg_test))
    print(classification_report(**arg_test))


print('Summary of Accuracy Score\n\n')
for i in models:
    model = models[i]
    print(i + ' Model: ',accuracy_score(Y_test, model.predict(X_test)).round(4))





# #### From the above accuracy summary, Logistic Regression, Random Forest and KNN models all gives high accuracy score of 0.94




# #### Cross Validation

from sklearn.model_selection import cross_val_score
gnb = GaussianNB()
scores = cross_val_score(gnb, X_train, Y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))



#model saving

import pickle
filename = r'Final_model_sv.sav'
pickle.dump(sv, open(filename,'wb'))



plt.bar(['Decission Tree','Logistic','KNN','random Forest','SVM'],[ac_dt,ac_lr,acc_knn,acc_rf,ac_sv,])
plt.xlabel("Alggorithams")
plt.ylabel("Accuracy")
plt.show()

#xgboost model
from xgboost import  XGBClassifier
xgc=XGBClassifier(objective='binary:logistic',n_estimators=100000,max_depth=5,learning_rate=0.001,n_jobs=-1)
xgc.fit(X_train,Y_train)
predict=xgc.predict(X_test)
print('Accuracy --> ',accuracy_score(predict,test_y))
print('F1 Score --> ',f1_score(predict,test_y))
print('Classification Report  --> \n',classification_report(predict,test_y))
print(predict)






