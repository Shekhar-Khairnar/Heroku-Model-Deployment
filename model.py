#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import pickle


# In[2]:


# Importing the dataset
dataset = pd.read_csv('c:/users/shekh/Desktop/GitProjects/Deploy/50_Startups.csv')
dataset.tail()


# In[3]:


#extracting numerical features
# list of numerical variables
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']


# In[7]:


# create an empty list to store the output indices from the multiple columns

index_list = []
for feature in numerical_features:
    index_list.extend(outliers(dataset,feature))


# In[8]:


# define a function called 'outliers' which returns a list of index of outliers
# IQR = Q3-Q1

def outliers(df,ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    ls = df.index[ (df[ft] < lower_bound) | (df[ft] > upper_bound)]
    
    return ls


# In[9]:


def remove(df,ls):
    ls =sorted(set(ls))
    df = df.drop(ls)
    return df


# In[10]:


dataset = remove(dataset,index_list)


# In[11]:


def zerovalues(df,ft):
    
    ls = df.index[ (df[ft] == 0.0)]
    
    return ls


# In[12]:


index_list1 = []
for feature in numerical_features:
    index_list1.extend(zerovalues(dataset,feature))


# In[13]:


# define a function called 'remove' which returns a cleaned dataframe without zerovalues

def remove1(df,ls):
    ls =sorted(set(ls))
    df = df.drop(ls)
    return df


# In[14]:


X = dataset.iloc[:,:-1] #independent features
y = dataset.iloc[:,-1] #dependent features


# In[15]:


states = pd.get_dummies(X['State'], drop_first = True)


# In[16]:


X = X.drop('State', axis = 1)


# In[17]:


X = pd.concat([X,states], axis = 1)


# In[18]:


X.head()


# In[19]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3,random_state = 0)


# In[20]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 700, random_state = 0)
regressor.fit(X_train,y_train)


# In[23]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[ ]:




