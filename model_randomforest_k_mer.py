#!/usr/bin/env python
# coding: utf-8

# # SVM Classifier with $k$-mer Encoding

# In[1]:


import os
import pickle
import numpy as np

# SADR: path to the dataset.
dataset_path = os.path.join("preprocessed_datasets", "dataset_k_mer.pkl")

# SADR: loading training data.
with open(dataset_path, "rb") as f:
    dataset_k_mer = pickle.load(f)

# SADR: getting the training, validation, and testing data.
X_train, y_train = dataset_k_mer["X_train"], dataset_k_mer["y_train"]
X_val, y_val = dataset_k_mer["X_val"], dataset_k_mer["y_val"]
X_test, y_test = dataset_k_mer["X_test"], dataset_k_mer["y_test"]

print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"X_val: {X_val.shape}")

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import KFold

X=np.concatenate([X_train,X_val])
y=np.concatenate([y_train,y_val])

maxdepths=np.array(range(15,35))
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

with open("random_forest_classifier_kmer.pkl", "wb") as f:
    pickle.dump(model, f)
    
importances = rf.feature_importances_

sorted_indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for i in sorted_indices[0:10]:
    print(f"Feature {i}: {importances[i]:.4f}")
    