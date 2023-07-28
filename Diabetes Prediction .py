#!/usr/bin/env python
# coding: utf-8

# # Importing dependencies

# In[12]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # Loading Dataset

# In[13]:


diabetes_dataset=pd.read_csv('diabetes.csv')


# In[14]:


diabetes_dataset.head()


# In[15]:


diabetes_dataset.shape


# In[16]:


# Getting statistical measures of data
diabetes_dataset.describe()


# In[17]:


diabetes_dataset['Outcome'].value_counts()


# In[19]:


diabetes_dataset.groupby('Outcome').mean()


# In[21]:


# Separating data and labels
# axis=1 indicates droping of column, axis=0 indicates droping of rows
X=diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y=diabetes_dataset['Outcome']


# In[22]:


print(X)


# In[23]:


print(Y)


# # Data standardization

# In[25]:


scaler=StandardScaler()


# In[26]:


scaler.fit(X)


# In[27]:


standardized_data=scaler.transform(X)


# In[28]:


print(standardized_data)


# In[29]:


X=standardized_data


# In[30]:


print(X)
print(Y)


# # Train Test split

# In[66]:


X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2) 


# In[67]:


print(X.shape,X_train.shape,X_test.shape)


# # Training the model

# In[68]:


classifier=svm.SVC(kernel='linear')


# In[69]:


# training svm classifier
classifier.fit(X_train,Y_train)


# # Model Evaluation

# In[70]:


#Accuracy score on training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[71]:


print('Training data accuracy = ',training_data_accuracy)


# In[72]:


#Accuracy score on test data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[73]:


print('Test data Accuracy = ',test_data_accuracy)


# # Making Prediction

# In[78]:


input_data=(4,200,125,0,0,37.6,0.621,50)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
prediction=classifier.predict(std_data)
print(prediction)

if (prediction[0]==0):
    print('The person is NOT DIABETIC')
else:
    print('The person is DIABETIC')


# In[ ]:




