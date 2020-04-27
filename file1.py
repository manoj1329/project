#!/usr/bin/env python
# coding: utf-8

# In[3]:


from mpl_toolkits.mplot3d import Axes3D #For Basic ploting
from sklearn.preprocessing import StandardScaler #Preprocessing
from sklearn import preprocessing    # Preprocessing
from sklearn.naive_bayes import GaussianNB #import gaussian naive bayes model
from sklearn.tree import DecisionTreeClassifier #import Decision tree classifier
from sklearn import metrics  #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#sklearn
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#load package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[4]:


import warnings

warnings.filterwarnings('ignore')


# In[5]:


data = 'C:\\Users\\DhanWin\\Desktop\\mmj\\ddos 1\\train_mosaic.csv'

train_data = pd.read_csv(data)


# In[6]:


data1 = 'C:\\Users\\DhanWin\\Desktop\\mmj\\ddos 1\\test_mosaic.csv'

test_data = pd.read_csv(data)


# In[7]:


train_data.shape,test_data.shape


# In[8]:


test_data.head()


# In[9]:


train_data.head()


# In[10]:


train_data['Label'].unique()
test_data['Label'].unique()


# In[11]:


# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
test_data['Label'] = label_encoder.fit_transform(test_data['Label'])


# In[12]:


train_data.head()


# In[13]:


test_data.head()


# In[17]:


x_train = train_data.drop('Label',axis=1)
x_test = test_data.drop('Label',axis=1)
y_train = train_data['Label']
y_test = test_data['Label']


# In[15]:


MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model. RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

#Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
   #tree.ExtraTreeClassifier(),
    
    ]


# In[ ]:


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)


row_index = 0
for alg in MLA:
    
    
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)
    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)





    row_index+=1
    
MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
MLA_compare

