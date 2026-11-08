# CSC525: Principles of Machine Learning
# Module 5 Portfolio Milestone
# Proposal Intake Assistant - Model Training
# Belinda Duncan
# October 18, 2026
#
# Purpose:
# Train a machine learning model to classify customer requests
# into three categories: Historian & Visualization, CONNECT /
# Cloud Integration, and Adapters / Data Connectivity.
# The program uses TF-IDF to convert the customer requests
# into numerical features and Logistic Regression to predict
# the classification. The results are evaluated using test
# data, and the trained model is saved for future use.
# _______________________________________________________________________________________
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib

# _______________________________________________________________________________________
# Locate the project folders
# The CSV is stored in the Training_Data folder, one level
# above the folder containing this Python script.
project_folder = Path(__file__).resolve().parent.parent
dataset_path = project_folder / "Training_Data" / "CSC525_Training_Dataset.csv"

# Read the customer requests and their assigned classifications.
df = pd.read_csv(dataset_path)

# Check that the file loaded and that each category has
# the expected number of customer requests.
print("CSC525 - Proposal Intake Assistant")
print("Dataset loaded successfully!")
print(f"Total customer requests: {len(df)}")

print("\nClassification distribution:")
print(df["classification_label"].value_counts())

print("\nFirst five customer requests:")
print(df.head())
# _______________________________________________________________________________________
# Split the dataset into training and testing

# X is the customer request, and y is the classification
# we assigned to that request in the training spreadsheet.
X = df["customer_request"]
y = df["classification_label"]

# Use 80% of the requests for training and hold back 20% to see how the model performs on requests it has not seen.
# Stratify keeps the three categories balanced in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Verify the split before moving into model training.
print("\nTraining and Testing Split:")
print(f"Training requests: {len(X_train)}")
print(f"Testing requests: {len(X_test)}")

print("\nTraining classification distribution:")
print(y_train.value_counts())

print("\nTesting classification distribution:")
print(y_test.value_counts())
# _______________________________________________________________________________________
# Convert customer text into numerical features
# Logistic Regression cannot work directly with sentences. TF-IDF converts the text into numerical values based on the words and phrases found in the customer requests. The ngram_range setting includes single words and two-word combinations.

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True
)

# Build the vocabulary using only the training requests. The test requests use that same vocabulary so they do not influence what the model learns during training.

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Review the number of features created from the text.
print("\nTF-IDF Feature Extraction:")
print(f"Training matrix shape: {X_train_tfidf.shape}")
print(f"Testing matrix shape: {X_test_tfidf.shape}")
print(f"Total features: {len(tfidf.get_feature_names_out())}")

print("\nFirst 20 vocabulary features:")
print(tfidf.get_feature_names_out()[:20])
# _______________________________________________________________________________________
# Train the Logistic Regression classifier
# Create the classification model using the selected settings.C controls regularization, while max_iter gives the model enough iterations to complete the training process.

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

# Train the model using the numerical features and the correct classifications from our training dataset.
model.fit(X_train_tfidf, y_train)

print("\nLogistic Regression Training:")
print("Model training completed successfully!")

print("\nClassification categories:")
print(model.classes_)

# Ask the trained model to classify the 18 test requests.
y_pred = model.predict(X_test_tfidf)
# Compare the first five predictions with their actual labels.
print("\nFirst five test predictions:")
for actual, predicted in zip(y_test.iloc[:5], y_pred[:5]):
    print(f"Actual: {actual} | Predicted: {predicted}")
# _______________________________________________________________________________________
# Evaluate the classification model
# Accuracy shows how many of the test requests were classified correctly.

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print(f"Accuracy: {accuracy:.2%}")

# Precision, recall, and F1-score give us a better picture of how the model performs within each classification.
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# The confusion matrix shows which categories were predicted correctly and where the model confused one category with another.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))

print("\nConfusion Matrix Label Order:")
print(model.classes_)

# Review each test request to see where the model made mistakes.These results will help identify areas for improvement.
print("\nDetailed Test Predictions:")

for request, actual, predicted in zip(X_test, y_test, y_pred):
    result = "CORRECT" if actual == predicted else "INCORRECT"

    print(f"\nCustomer Request: {request}")
    print(f"Actual Classification: {actual}")
    print(f"Predicted Classification: {predicted}")
    print(f"Result: {result}")

# _______________________________________________________________________________________
# Save the trained model 
# Store the trained model and TF-IDF vectorizer in a separate folder so they can be used later by the chatbot application.

model_folder = project_folder / "Model_Training" / "saved_models"
model_folder.mkdir(parents=True, exist_ok=True)

# Both files are needed to classify new customer requests.The vectorizer prepares the text, and the model predicts which classification the request belongs to.
joblib.dump(model, model_folder / "logistic_regression_model.joblib")
joblib.dump(tfidf, model_folder / "tfidf_vectorizer.joblib")

print("\nModel files saved successfully!")
print(f"Location: {model_folder}")