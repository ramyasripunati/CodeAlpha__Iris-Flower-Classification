# Project 2: Iris Flower Classification Using AI

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
print("[Step 1] Loading Iris Flower Dataset...\n")
iris = load_iris()
# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
# Add target column
df["target"] = iris.target
print("Dataset Shape:", df.shape)
print("\nFlower Categories:")
print(iris.target_names)
print("\nFirst 5 Rows:")
print(df.head())

# Split Dataset
print("\n[Step 2] Splitting Dataset...\n")
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Model
print("\n[Step 3] Training KNN Model...\n")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("Model Training Completed!")

# Predictions and Evaluation
print("\n[Step 4] Predicting Test Data...\n")
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))