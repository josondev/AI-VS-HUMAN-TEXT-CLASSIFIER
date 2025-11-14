# FastAPI for AI Text Detector - Modern, Fast, Production-Ready

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from detector import AITextDetector
import os
from typing import Optional

# Initialize FastAPI
app = FastAPI(
    title="AI Text Detector API",
    description="Detect if text is AI-generated or human-written using Naive Bayes",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
detector = AITextDetector()
MODEL_DIR = 'model'

@app.on_event("startup")
async def load_model():
    """Load model when API starts."""
    try:
        if detector.model_exists(MODEL_DIR):
            detector.load_model(MODEL_DIR)
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No model found - train first!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


# Request model
class PredictRequest(BaseModel):
    text: str

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


# Response model
class PredictResponse(BaseModel):
    label: str
    ai_probability: float
    human_probability: float
    confidence: str
    certainty: float
    text_length: int
    word_count: int


# Health check response
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/")
def root():
    """API information."""
    return {
        "name": "AI Text Detector API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": detector.is_trained,
        "endpoints": {
            "POST /predict": "Classify text as AI or Human",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if detector.is_trained else "unhealthy",
        "model_loaded": detector.is_trained
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Classify text as AI-generated or human-written.

    **Request Body:**
    - text (str): The text to classify

    **Response:**
    - label: "AI" or "Human"
    - ai_probability: Probability text is AI-generated (0-1)
    - human_probability: Probability text is human-written (0-1)
    - confidence: Confidence level (very_high, high, medium, low)
    - certainty: Absolute difference between probabilities (0-1)
    - text_length: Character count
    - word_count: Word count
    """
    try:
        if not detector.is_trained:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Predict
        label, ai_prob, human_prob = detector.predict(request.text)

        # Calculate confidence
        certainty = abs(ai_prob - human_prob)
        if certainty > 0.8:
            confidence = "very_high"
        elif certainty > 0.5:
            confidence = "high"
        elif certainty > 0.2:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "label": label,
            "ai_probability": round(ai_prob, 4),
            "human_probability": round(human_prob, 4),
            "confidence": confidence,
            "certainty": round(certainty, 4),
            "text_length": len(request.text),
            "word_count": len(request.text.split())
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
