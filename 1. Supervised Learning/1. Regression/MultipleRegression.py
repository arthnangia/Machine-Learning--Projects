#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# - In this project, a regression model is developed to predict the probability of being accepted for Graduate school.
# - Citation: Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019
# 
# - The dataset contains the following parameters: 
#     - GRE Scores ( out of 340 ) 
#     - TOEFL Scores ( out of 120 ) 
#     - University Rating ( out of 5 ) 
#     - Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 
#     - Undergraduate GPA ( out of 10 ) 
#     - Research Experience ( either 0 or 1 ) 
#     - Chance of Admit ( ranging from 0 to 1 )

# # STEP #1: IMPORT LIBRARIES 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # STEP #2: IMPORT DATASET

# In[2]:


admission_df = pd.read_csv('C:/ConsoleFlare/Machine Learning Course/Data/Admission.csv')


# In[3]:


admission_df.info()


# In[4]:


admission_df.describe()


# In[5]:


admission_df


# # STEP #3: VISUALIZE DATASET

# In[14]:


plt.figure(figsize = (10, 10))
sns.heatmap(admission_df.corr(), annot = True)


# In[15]:


sns.pairplot(admission_df)


# In[17]:


admission_df.isnull()


# In[16]:


sns.heatmap(admission_df.isnull())


# # STEP #4: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# In[18]:


X = admission_df.iloc[:, :-1].values


# In[19]:


y = admission_df.iloc[:, -1].values


# In[23]:


print(y)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 0)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# # STEP #5: TRAINING THE MODEL

# In[27]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[28]:


y_predict = regressor.predict(X_test)


# # STEP #6: EVALUATING THE MODEL 

# In[29]:


plt.scatter(y_test, y_predict, color = 'r')
plt.ylabel('Model Predictions')
plt.xlabel('True (ground truth)')


# In[30]:


print(X_train.shape)


# # STEP #7: Eliminating Columns using Backward Elimination

# In[31]:


import statsmodels.api as smf  
# add a column of ones as integer data type 
X_train = np.append(arr = np.ones((320,1)).astype(int), values=X_train, axis=1)    
# choose a Significance level usually 0.05, if p>0.05 
#  for the highest values parameter, remove that value 

regressor_OLS=smf.OLS(endog = y_train, exog=X_train).fit()  
regressor_OLS.summary()  


# In[32]:


print(X_train)


# In[ ]:




