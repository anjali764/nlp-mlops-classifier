# app/main.py (Complete code from previous response)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os


# ... (the rest of the API code from previous response)

class SentenceInput(BaseModel):
    sentence: str


# --- Initialization ---
app = FastAPI(title="NLP Sentence Classifier API", version="1.0")
model = None
vectorizer = None


# --- Startup Event: Load Model Artifacts ---
@app.on_event("startup")
async def load_artifacts():
    global model, vectorizer
    model_path = "model/classifier.pkl"
    vectorizer_path = "model/vectorizer.pkl"
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        # This will be run *inside* the Docker container later, where 'model/' should be copied.
        print("Model files not found, but continuing for now. Check Docker copy steps.")
        return
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and Vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # Not strictly necessary to raise here if we handle 500 in the predict method


# --- Prediction Endpoint ---
@app.post("/predict")
async def predict_correctness(data: SentenceInput):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model artifacts failed to load on startup.")

    input_text = [data.sentence]
    input_vectorized = vectorizer.transform(input_text)
    prediction = model.predict(input_vectorized)[0]
    result = "Grammatically Correct" if prediction == 1 else "Grammatically Incorrect"

    return {
        "sentence": data.sentence,
        "prediction_label": result,
        "prediction_score": float(model.predict_proba(input_vectorized)[0][prediction])
    }


# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}