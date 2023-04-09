#!/usr/bin/env python
# coding: utf-8

# #### Name: Navya N
# #### Usn   : 20BTRCD053

# In[ ]:





# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df_train=pd.read_csv("file:///C:/Users/Navya/Downloads/train.csv")
df_train


# In[7]:


df_train.head()


# In[8]:


df_train.shape


# In[9]:


df_train.info()


# In[10]:


df_train.columns


# In[11]:


df_train.describe()


# In[12]:


df_train.isnull().sum()


# In[13]:


missing = df_train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
plt.figure(figsize=(8,4))
missing.plot.bar()


# In[14]:


sns.set(rc={'figure.figsize' :(8,4)})
sns.distplot(df_train['SalePrice'],bins=20)


# In[ ]:





# ### Correlation

# In[15]:


numeric = df_train.select_dtypes(include=[np.number])


# In[16]:


categorical = df_train.select_dtypes(include=[np.object])


# In[17]:


from sklearn.preprocessing import OrdinalEncoder
OE = OrdinalEncoder()
for i in categorical.columns:
    df_train[[i]]=OE.fit_transform(df_train[[i]])


# In[18]:


correlation = numeric.corr()
print(correlation['SalePrice'].sort_values(ascending=False),'\n')


# In[19]:


f,ax=plt.subplots(figsize=(10,8))
sns.heatmap(correlation,annot=True,square=True, vmax=0.8)


# In[20]:


k = 11
cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index
print(cols)
cm = np.corrcoef(df_train[cols].values.T)
f, ax = plt.subplots(figsize=(14,12))
sns.heatmap(cm, vmax=0.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels= cols.values, annot_kws={'size':12},yticklabels = cols.values)


# ### Scatterplot

# In[21]:


sns.scatterplot(x=df_train['GarageCars'],y=df_train['SalePrice'])


# In[22]:


fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(14,10))

sns.regplot(x=df_train['OverallQual'],y=df_train['SalePrice'], scatter = True, fit_reg= True, ax = ax1)

sns.regplot(x=df_train['GrLivArea'],y=df_train['SalePrice'], scatter = True, fit_reg= True, ax = ax2)

sns.regplot(x=df_train['GarageArea'],y=df_train['SalePrice'], scatter = True, fit_reg= True, ax = ax3)

sns.regplot(x=df_train['FullBath'],y=df_train['SalePrice'], scatter = True, fit_reg= True, ax = ax4)

sns.regplot(x=df_train['YearBuilt'],y=df_train['SalePrice'], scatter = True, fit_reg= True, ax = ax5)

sns.regplot(x=df_train['WoodDeckSF'],y=df_train['SalePrice'], scatter = True, fit_reg= True,ax = ax6)


#  ### BoxPlot

# In[23]:


sns.boxplot(x=df_train["SalePrice"])


# In[24]:


f, ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x=df_train['SaleType'],y=df_train["SalePrice"])
fig.axis(ymin=0,ymax=800000);
xt = plt.xticks(rotation=45)


# In[25]:


f, ax = plt.subplots(figsize=(16,10))
fig = sns.boxplot(x=df_train['OverallQual'],y=df_train["SalePrice"])
fig.axis(ymin=0,ymax=800000);


# ## Removal of Outliers

# In[ ]:





# In[26]:


df_train['SalePrice'].describe()


# In[27]:


sns.boxplot(x=df_train["SalePrice"])


# In[28]:


df_train.shape


# In[29]:


first_quartile = df_train['SalePrice'].quantile(.25)
third_quartile = df_train['SalePrice'].quantile(.75)
IQR=third_quartile-first_quartile


# In[30]:


new_boundary = third_quartile + 3*IQR


# In[31]:


df_train.drop(df_train[df_train['SalePrice']>new_boundary].index,axis=0,inplace=True)


# In[32]:


df_train.shape


# In[33]:


df_train.fillna(method="bfill",inplace =True)


# ## Removing Unnecessary Features

# In[34]:


columns_to_remove=['BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath',
                   'BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath',
                   'MiscVal','Id','LowQualFinSF','YrSold', 'OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']


# In[35]:


df_train.shape


# In[36]:


x = df_train.drop('SalePrice',axis = 1)
y = df_train.SalePrice


# In[48]:


y.info()


# In[50]:


x.info()


# In[44]:


x.isnull().sum().sum()


# In[45]:


x.dropna(inplace=True)


# In[53]:


y=y[:1412]


# In[54]:


from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy = train_test_split(x,y)


# In[55]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[56]:


lr.fit(trainx,trainy)


# In[57]:


yp = lr.predict(testx)


# In[58]:


from sklearn.metrics import r2_score
r2_score(testy,yp)


# In[ ]:





# In[ ]:





# In[59]:


df_train.dropna(inplace = True)


# In[60]:


for i in df_train.columns:
    df_train[i] = df_train[i].astype("int")


# In[ ]:





# In[61]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[62]:


rf.fit(trainx,trainy)


# In[63]:


ypr = rf.predict(testx)


# In[64]:


r2_score(testy,ypr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




