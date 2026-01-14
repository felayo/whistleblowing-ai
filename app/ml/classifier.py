"""
TfidfVectorizer → turns sentences into smart numbers 
LogisticRegression → the actual classification brain
clean_text → my custom function that removes noise such as punctuation, extra spaces, stopwords, lowercase etc
"""
import os
import joblib
from app.ml.preprocess import clean_text

# Paths for saved model & vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ml_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Load model & vectorizer once at startup
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

def classify_report(report_text: str):
    """
    Classifies a whistleblower report into one of the 13 categories.
    
    Args:
        report_text (str): The report text from the whistleblower.
    
    Returns:
        tuple: (predicted_category (str), confidence (float))
    """
    # 1. Preprocess text
    clean = clean_text(report_text)
    
    # 2. Vectorize
    vect_text = vectorizer.transform([clean])
    
    # 3. Predict probabilities
    pred_probs = model.predict_proba(vect_text)[0]
    
    # 4. Get highest probability & corresponding category
    pred_index = pred_probs.argmax()
    predicted_category = model.classes_[pred_index]
    confidence = float(pred_probs[pred_index])
    
    return predicted_category, confidence
