#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# In[2]:


data = pd.read_csv('weather.csv')
data.head()


# In[3]:


print(data.info())


# In[4]:


print(data.describe())


# In[5]:


data.columns


# In[6]:


data.dtypes


# In[7]:


data.drop('Unnamed: 0', axis=1, inplace=True)


# In[8]:


data.dropna(inplace=True)


# In[9]:


data['Date'] = pd.to_datetime(data['Date'])


# In[10]:


categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])


# In[11]:


X = data.drop('RainTomorrow', axis=1) 
y = data['RainTomorrow']  


# In[12]:


numeric_cols=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                'Temp9am', 'Temp3pm', 'RISK_MM']
X[numeric_cols]=(X[numeric_cols]-X[numeric_cols].mean())/X[numeric_cols].std()


# In[13]:


numeric_cols=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                'Temp9am', 'Temp3pm', 'RISK_MM']
numeric_data=data[numeric_cols]
corr_matrix=numeric_data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[14]:


plt.figure(figsize=(12,10))
for i, col in enumerate(numeric_cols):
    plt.subplot(4,5,i+1)
    sns.histplot(data[col],kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()


# In[15]:


data['Year']=pd.to_datetime(data['Date']).dt.year
data['Month']=pd.to_datetime(data['Date']).dt.month
data['Day']=pd.to_datetime(data['Date']).dt.day


# In[16]:


data['TempDiff']=data['MaxTemp'] - data['MinTemp']


# In[17]:


imputer=SimpleImputer(strategy='mean')
numeric_cols=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                'Temp9am', 'Temp3pm', 'RISK_MM']
data[numeric_cols]=imputer.fit_transform(data[numeric_cols])


# In[18]:


skewed_features=['Rainfall', 'Evaporation']
data[skewed_features]=np.log1p(data[skewed_features])
data.drop([ 'Date'], axis=1,inplace=True)
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
data=pd.get_dummies(data,columns=categorical_cols,drop_first=True)
X=data.drop('RainTomorrow',axis=1)  
y=data['RainTomorrow'] 


# In[19]:


imputer=SimpleImputer(strategy='mean')
X_imputed=imputer.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_imputed, y,test_size=0.2,random_state=42)


# In[20]:


rf_classifier=RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


y_pred=rf_classifier.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)
confusion_mat=confusion_matrix(y_test, y_pred)


# In[22]:


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)
print("Confusion Matrix:")
print(confusion_mat)


# In[31]:


X = np.arange(1,11)
Y = np.random.randint(10,100, size=10)
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
plt.plot(x, y, marker='o', linestyle='-', color='blue', label='Line')
plt.title('Line Graph')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()


# In[27]:


plt.figure(figsize=(8, 4))
plt.bar(x, y, color='green')
plt.title('Bar Graph')
plt.xlabel('X-axis')
plt.ylabel('Values')
plt.show()


# In[28]:


plt.figure(figsize=(8, 4))
plt.scatter(x, y, color='red', s=100, edgecolors='black')
plt.title('Scatter Plot (Points)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# In[30]:





# In[ ]:




