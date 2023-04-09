#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("heartDisease.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df.info()


# In[20]:


df.describe()


# In[11]:


corr=df.corr()


# In[12]:


corr


# In[14]:


df.isnull().sum()


# In[15]:


df.isnull().sum().sum()


# In[16]:


sns.set(rc={'figure.figsize' :(8,4)})
sns.distplot(df['target'],bins=20)


# In[ ]:





# In[21]:


sns.countplot(x='target', data=df)
plt.show()


# In[22]:


sns.displot(df['age'], kde=False)
plt.show()
sns.displot(df['trestbps'], kde=False)
plt.show()
sns.displot(df['chol'], kde=False)
plt.show()
sns.displot(df['thalach'], kde=False)
plt.show()
sns.displot(df['oldpeak'], kde=False)
plt.show()


# In[24]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()


# ### Distribution OF Categorial Variables

# In[26]:


sns.countplot(x='sex', data=df)
plt.show()
sns.countplot(x='cp', data=df)
plt.show()
sns.countplot(x='fbs', data=df)
plt.show()
sns.countplot(x='restecg', data=df)
plt.show()
sns.countplot(x='exang', data=df)
plt.show()
sns.countplot(x='slope', data=df)
plt.show()
sns.countplot(x='ca', data=df)
plt.show()
sns.countplot(x='thal', data=df)
plt.show()


# In[ ]:





# ## Logistic Regression

# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)


# In[30]:


lr = LogisticRegression()


# In[31]:


lr.fit(X_train, y_train)


# In[32]:


yp=lr.predict(X_test)


# In[34]:


accuracy = accuracy_score(y_test, yp)
print(accuracy)


# In[ ]:





# ## KNN

# In[35]:


from sklearn.neighbors import KNeighborsClassifier


# In[36]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[37]:


knn.fit(X_train, y_train)


# In[38]:


yp = knn.predict(X_test)


# In[40]:


accuracy = accuracy_score(y_test, yp)
print(accuracy)


# In[ ]:





# ## Decisioon Tree

# In[42]:


from sklearn.tree import DecisionTreeClassifier


# In[43]:


dt = DecisionTreeClassifier()


# In[44]:


dt.fit(X_train, y_train)


# In[45]:


yp = dt.predict(X_test)


# In[47]:


accuracy = accuracy_score(y_test, yp)
print(accuracy)


# In[ ]:





# ## Random Forest

# In[48]:


from sklearn.ensemble import RandomForestClassifier


# In[49]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)


# In[51]:


rf.fit(X_train, y_train)


# In[53]:


yp = rf.predict(X_test)


# In[54]:


accuracy = accuracy_score(y_test, yp)
print(accuracy)


# In[ ]:





# ## XGBoost

# In[56]:


import xgboost as xgb 


# In[57]:


xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05)


# In[58]:


xgb_model.fit(X_train, y_train)


# In[59]:


yp = xgb_model.predict(X_test)


# In[60]:


accuracy = accuracy_score(y_test, yp)
print(accuracy)


# In[ ]:




