import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "intent_model.pkl")

def train_production_model():
    """Builds and saves the full pipeline (Vectorize + Classify)."""
    data_path = os.path.join(BASE_DIR, "queries_dataset.csv")
    df = pd.read_csv(data_path)
    
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', LinearSVC())
    ])

    print(" Training the 10-Intent Brain...")
    model_pipeline.fit(df["Query"], df["Intent"])
    
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f" Production model saved to {MODEL_PATH}")

def predict_intent(query):
    """Loads the pipeline and predicts the intent of a string."""
    if not os.path.exists(MODEL_PATH):
        return "Model not trained"
    
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([query])
    return prediction[0]

if __name__ == "__main__":
    train_production_model()