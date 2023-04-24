#!/usr/bin/env python
# coding: utf-8

# ### Comparitive chart of all models built in experiment 05

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('heartDisease.csv')
df.head(5)


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.nunique(axis=0)


# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# no null values are present

# In[8]:


df.duplicated().sum()


# In[9]:


df.drop_duplicates(inplace=True)


# duplicate values have been handled

# In[10]:


df['target'].value_counts()


# In[11]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)  


# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ### Logistic Regression

# In[14]:


from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=42) 
model1.fit(x_train, y_train)

y_pred1 = model1.predict(x_test) 
print(classification_report(y_test, y_pred1))

log = model1.score(x_test, y_test)


# In[15]:


tp = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred1[i] == 1])
fp = sum([1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred1[i] == 1])
fn = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred1[i] == 0])

precision1 = tp / (tp + fp)
recall1 = tp / (tp + fn)
f1_score1 = 2 * (precision1 * recall1) / (precision1 + recall1)

# Print the results
print("Precision:", precision1)
print("Recall:", recall1)
print("F1 Score:", f1_score1)


# ### K-NN (K-Nearest Neighbors)

# In[16]:


from sklearn.metrics import classification_report 
from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier() 
model2.fit(x_train, y_train)  

y_pred2 = model2.predict(x_test) 
print(classification_report(y_test, y_pred2)) 

knn = model2.score(x_test, y_test)


# In[17]:


tp = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred2[i] == 1])
fp = sum([1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred2[i] == 1])
fn = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred2[i] == 0])

precision2 = tp / (tp + fp)
recall2 = tp / (tp + fn)
f1_score2 = 2 * (precision2 * recall2) / (precision2 + recall2)

# Print the results
print("Precision:", precision2)
print("Recall:", recall2)
print("F1 Score:", f1_score2)


# ### SVM (Support Vector Machine)

# In[18]:


from sklearn.metrics import classification_report 
from sklearn.svm import SVC

model3 = SVC(random_state=42) 
model3.fit(x_train, y_train) 

y_pred3 = model3.predict(x_test)
print(classification_report(y_test, y_pred3))

svm = model3.score(x_test, y_test)


# In[19]:


tp = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred3[i] == 1])
fp = sum([1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred3[i] == 1])
fn = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred3[i] == 0])

precision3 = tp / (tp + fp)
recall3 = tp / (tp + fn)
f1_score3 = 2 * (precision3 * recall3) / (precision3 + recall3)

# Print the results
print("Precision:", precision3)
print("Recall:", recall3)
print("F1 Score:", f1_score3)


# ### Naives Bayes Classifier

# In[20]:


from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB

model4 = GaussianNB() 
model4.fit(x_train, y_train) 

y_pred4 = model4.predict(x_test) 
print(classification_report(y_test, y_pred4)) 

nb = model4.score(x_test, y_test)


# In[21]:


tp = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred4[i] == 1])
fp = sum([1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred4[i] == 1])
fn = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred4[i] == 0])

precision4 = tp / (tp + fp)
recall4 = tp / (tp + fn)
f1_score4 = 2 * (precision4 * recall4) / (precision4 + recall4)

# Print the results
print("Precision:", precision4)
print("Recall:", recall4)
print("F1 Score:", f1_score4)


# ### Decision Trees

# In[22]:


from sklearn.metrics import classification_report 
from sklearn.tree import DecisionTreeClassifier

model5 = DecisionTreeClassifier(random_state=42) 
model5.fit(x_train, y_train) 

y_pred5 = model5.predict(x_test)
print(classification_report(y_test, y_pred5)) 

dt = model5.score(x_test, y_test)


# In[23]:


tp = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred5[i] == 1])
fp = sum([1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred5[i] == 1])
fn = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred5[i] == 0])

precision5 = tp / (tp + fp)
recall5 = tp / (tp + fn)
f1_score5 = 2 * (precision5 * recall5) / (precision5 + recall5)

# Print the results
print("Precision:", precision5)
print("Recall:", recall5)
print("F1 Score:", f1_score5)


# ### Random Forest

# In[24]:


from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=42)
model6.fit(x_train, y_train)

y_pred6 = model6.predict(x_test) 
print(classification_report(y_test, y_pred6)) 

rf = model6.score(x_test, y_test)


# In[25]:


tp = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred6[i] == 1])
fp = sum([1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred6[i] == 1])
fn = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred6[i] == 0])

precision6 = tp / (tp + fp)
recall6 = tp / (tp + fn)
f1_score6 = 2 * (precision6 * recall6) / (precision6 + recall6)

# Print the results
print("Precision:", precision6)
print("Recall:", recall6)
print("F1 Score:", f1_score6)


# ### XGBoost

# In[26]:


from xgboost import XGBClassifier

model7 = XGBClassifier(random_state=42)
model7.fit(x_train, y_train)

y_pred7 = model7.predict(x_test)
print(classification_report(y_test, y_pred7))

xgb = model7.score(x_test, y_test)


# In[27]:


tp = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred7[i] == 1])
fp = sum([1 for i in range(len(y_test)) if y_test[i] == 0 and y_pred7[i] == 1])
fn = sum([1 for i in range(len(y_test)) if y_test[i] == 1 and y_pred7[i] == 0])

precision7 = tp / (tp + fp)
recall7 = tp / (tp + fn)
f1_score7 = 2 * (precision7 * recall7) / (precision7 + recall7)

# Print the results
print("Precision:", precision7)
print("Recall:", recall7)
print("F1 Score:", f1_score7)


# In[30]:


model = pd.DataFrame({'Model': ['Logistic Regression', 'KNN','SVM','Naive bayes Classifier','Decision Tree', 'Random Forest', 'XGB Regressor'], 'Score': [log, knn, svm, nb, dt, rf, xgb],'Model No.':[1,2,3,4,5,6,7],
                     'Precision':[precision1, precision2, precision3, precision4, precision5, precision6, precision7],
                     'Recall': [recall1, recall2, recall3, recall4, recall5, recall6, recall7],
                     'F1-Score': [f1_score1, f1_score2, f1_score3, f1_score4, f1_score5, f1_score6, f1_score7]})
models = model.sort_values(by = 'Score', ascending = False)
models = models.set_index('Model No.')
models

