import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def train_classifiers():
    # 1. Load the dataset you generated
    data_path = os.path.join(os.path.dirname(__file__), "queries_dataset.csv")
    if not os.path.exists(data_path):
        print("Error: queries_dataset.csv not found. Run generator.py first.")
        return

    df = pd.read_csv(data_path)
    X = df['Query']
    y = df['Intent']

    # 2. Vectorize the text (Convert words into mathematical weights)
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X_vectors = vectorizer.fit_transform(X)

    # Split data (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

    # 3. Train Support Vector Machine (SVM)
    print("\n--- Training SVM ---")
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds) * 100:.2f}%")
    
    # 4. Train Naive Bayes
    print("\n--- Training Naive Bayes ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_preds) * 100:.2f}%")

    # 5. Save the winning model and the vectorizer
    # (We are saving SVM here as our default "winner" for the backend to use later)
    model_path = os.path.join(os.path.dirname(__file__), "intent_model.pkl")
    vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
    
    joblib.dump(svm_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"\n✅ Training Complete! Model and Vectorizer saved to disk.")

if __name__ == "__main__":
    train_classifiers()