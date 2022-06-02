#!/usr/bin/env python
# coding: utf-8

# In[66]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import sklearn 

titanic_data=pd.read_csv("C:/Users/konda/Downloads/Data_Science_DataSets/titanic_logistic_regression.csv")
# print(len(titanic_data))-418
print(titanic_data.head(10))


# In[13]:


#Analyzing data
# sns.countplot(x="Survived", data=titanic_data)
sns.countplot(x="Survived", hue="Pclass",data=titanic_data)


# In[15]:


sns.countplot(x="SibSp",data=titanic_data)


# In[16]:


print(titanic_data.info())


# In[18]:


sns.countplot(x="PassengerId",hue="Survived",data=titanic_data)


# In[26]:


sns.countplot(x="Survived",hue="Sex",data=titanic_data)


# # #Data Wrangling

# In[28]:


titanic_data.isnull()


# In[29]:


print(titanic_data.isnull().sum())


# In[30]:


titanic_data.drop("Cabin",axis=1,inplace=True)


# In[31]:


titanic_data.head(5)


# In[32]:


titanic_data.dropna(inplace=True)


# In[33]:


sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)


# In[34]:


print(titanic_data.isnull().sum())


# In[35]:


titanic_data.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[36]:


print(titanic_data.head(5))


# In[37]:


sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[44]:


embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark.head(5)


# In[48]:


pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
pcl.head(5)


# In[53]:


titanic_data=pd.concat([titanic_data,embark,pcl],axis=1 )
titanic_data.head(5)


# In[54]:


x=titanic_data["Sex"]
x.head(4)


# In[57]:


s=pd.get_dummies(titanic_data['Sex'],drop_first=True)
s.head(5)


# In[58]:


titanic_data=pd.concat([titanic_data,s],axis=1 )
titanic_data.head(5)


# In[59]:


titanic_data.drop("Sex",inplace=True,axis=1)


# In[60]:


titanic_data.head(5)


# In[61]:


titanic_data.drop(["Pclass","Embarked"],inplace=True,axis=1)


# In[62]:


titanic_data.head(5)


# # Training data

# In[63]:


x=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]


# In[70]:


import sklearn 
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=1)


# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


logmodel=LogisticRegression()


# In[73]:


logmodel.fit(x_train,y_train)


# In[74]:


predictions= logmodel.predict(x_test)


# In[76]:


from sklearn.metrics import classification_report


# In[77]:


classification_report(y_test,predictions)


# # To find the accuracy- printing confusion matrix

# In[79]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[ ]:




