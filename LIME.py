# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:13:52 2019

@author: jayasans4085
"""
## https://github.com/marcotcr/lime/issues/166
## https://github.com/marcotcr/lime/issues/175
## https://github.com/marcotcr/lime/issues/73

## https://www.guru99.com/scikit-learn-tutorial.html

import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
#from __future__ import print_function

data = np.genfromtxt(r'C:/Users/jayasans4085/Downloads/LIME for Categorical/adult.data', delimiter=',', dtype='<U20')
feature_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status","Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss","Hours per week", "Country"]

#%%
labels = data[:,14]
le= sklearn.preprocessing.LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
class_names = le.classes_
data = data[:,:-1]

categorical_features = [1,3,5, 6,7,8,9,13]

categorical_names = {}
for feature in categorical_features:
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(data[:, feature])
    data[:, feature] = le.transform(data[:, feature])
    categorical_names[feature] = le.classes_
    
#%%
data = data.astype(float)
encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)

np.random.seed(1)
train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)

encoder.fit(data)
encoded_train = encoder.transform(train)

import xgboost
gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
gbtree.fit(encoded_train, labels_train)

sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))

predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)

#%%
explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = feature_names,class_names=class_names,
                                                   categorical_features=categorical_features, 
                                                   categorical_names=categorical_names, kernel_width=3)

np.random.seed(1)
i = 1653
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)

i = 10
exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
exp.show_in_notebook(show_all=False)

import pandas as pd
exp = pd.DataFrame(exp.as_list())
