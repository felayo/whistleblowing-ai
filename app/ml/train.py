import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_text

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "whistleblower_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# 1️⃣ Load dataset
df = pd.read_csv(DATASET_PATH)

# Check dataset structure
print("Dataset sample:")
print(df.head())

# 2️⃣ Preprocess text
print("Preprocessing text...")
df["clean_text"] = df["Description"].apply(clean_text)

# 3️⃣ Vectorization (TF-IDF)
print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["Category"]

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Train model
print("Training Naive Bayes classifier...")
model = MultinomialNB()
model.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7️⃣ Save model & vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECT_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Vectorizer saved to {VECT_PATH}")
