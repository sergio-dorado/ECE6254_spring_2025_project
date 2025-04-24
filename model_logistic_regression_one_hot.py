
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Portable path 
dataset_path = os.path.join("preprocessed_datasets", "dataset_one_hot.pkl")

# Load the dataset from local disk
with open(dataset_path, "rb") as f:
    dataset_one_hot = pickle.load(f)

# Extract train/val/test sets
X_train = dataset_one_hot["X_train"]
y_train = dataset_one_hot["y_train"]
X_val = dataset_one_hot["X_val"]
y_val = dataset_one_hot["y_val"]
X_test = dataset_one_hot["X_test"]
y_test = dataset_one_hot["y_test"]

# Use a solver optimized for large data (e.g., 'saga') and set higher max_iter
model = LogisticRegression(max_iter=800, solver='saga', verbose=1)
model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(classification_report(y_val, y_val_pred, target_names=["Animal", "Human"], digits=4))

# Evaluate on test set
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, y_test_pred, target_names=["Animal", "Human"], digits=4))