#!/usr/bin/env python
# coding: utf-8

# # SVM Classifier with One-Hot Encoding

# In[15]:


import os
import pickle
import numpy as np

# SADR: path to the dataset.
dataset_path = os.path.join("preprocessed_datasets", "dataset_one_hot_pca.pkl")

# SADR: loading training data.
with open(dataset_path, "rb") as f:
    dataset_one_hot = pickle.load(f)

# SADR: getting the training, validation, and testing data.
X, y = dataset_one_hot["Z_train"], dataset_one_hot["y_train"]
X_test, y_test = dataset_one_hot["Z_test"], dataset_one_hot["y_test"]

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import KFold

maxdepths=np.array(range(1,35))
score=0
model=[]

# Create custom F1 scorer (binary classification by default uses 'binary' average)
f1_scorer = make_scorer(f1_score, average='binary')

# Define KFold strategy
kf = KFold(n_splits=5, shuffle=True)

for i in maxdepths:
    
    rf = RFC(n_estimators=100, max_depth=i)
    rf.fit(X, y)
    scorenew = np.mean(CVS(rf, X, y, cv=kf, scoring=f1_scorer))
    if scorenew>score:
        score=scorenew
        model=rf
tree_depths = [tree.tree_.max_depth for tree in model.estimators_]
optimal_maxdepth=max(tree_depths)

print("Maximum tree depth:", max(tree_depths))
print("Accuracy k-fold:", score)

accuracy_score=accuracy_score(model.predict(X_test),y_test)
print("Accuracy test:", accuracy_score)
f1_score=f1_score(model.predict(X_test),y_test)
print("F1 test:", f1_score)
precision_score=precision_score(model.predict(X_test),y_test)
print("Precision test:", precision_score)
recall_score=recall_score(model.predict(X_test),y_test)
print("Recall test:", recall_score)

with open("random_forest_classifier_onehot.pkl", "wb") as f:
    pickle.dump(model, f)
    
importances = rf.feature_importances_

sorted_indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for i in sorted_indices[0:10]:
    print(f"Feature {i}: {importances[i]:.4f}")