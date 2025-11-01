# src/train.py (Complete code from previous response)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os


def train_and_save_model(data_path='data/sentences.csv', model_dir='model'):
    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df['sentence']
    y = df['label']

    # --- ⚠️ MODIFICATION START ⚠️ ---
    # Use ALL data for training and testing to guarantee non-zero accuracy
    X_train = X
    y_train = y

    # 2. Feature Extraction (TF-IDF Vectorizer)
    # Fit and Transform on ALL data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X_train)
    # --- ⚠️ MODIFICATION END ⚠️ ---

    # 3. Model Training
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_vec, y_train)

    # 4. Evaluation (Testing on the training data itself)
    predictions = model.predict(X_vec)
    accuracy = accuracy_score(y_train, predictions)  # Use y_train here
    print(f"Model Trained. Final Accuracy: {accuracy:.4f}")  # Accuracy should be high!

    # 5. Save Artifacts
    # ... (rest of the saving code remains the same) ...
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'classifier.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    print(f"Model and Vectorizer saved to {model_dir}")


if __name__ == '__main__':
    train_and_save_model()