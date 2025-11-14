# AI Text Detector - Deployment Ready

## 📁 Project Structure
```
.
├── detector.py          # Main detector class
├── fastapi_app.py              # Flask API
├── requirements.txt    # Dependencies
├── render.yaml         # Render config
├── Procfile           # Heroku config
└── model/             # Trained model (create this!)
    ├── vectorizer.joblib
    ├── classifier.joblib
    └── config.joblib
```

## 🚀 Quick Start

### 1. Train Model Locally
```bash
python detector.py
# Select option 1 to train
# Creates model/ folder with .joblib files
```

### 2. Test Locally
```bash
# CLI mode
python detector.py

# API mode
python app.py
# Visit http://localhost:5000
```

### 3. Deploy to Render
1. Push to GitHub
2. Connect to Render
3. Deploy (auto-detects render.yaml)
4. Done! ✅

## 🌐 API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "label": "AI",
  "ai_probability": 0.8523,
  "human_probability": 0.1477,
  "confidence": "high",
  "certainty": 0.7046,
  "text_length": 32,
  "word_count": 4
}
```

## 🧪 Test API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning is cool"}'
```

## 📦 Deployment Platforms

- **Render** (Recommended): Free tier, easy setup
- **Railway**: $5 credit/month
- **Heroku**: Uses Procfile
- **Vercel**: May have size limits

## ⚠️ Important

**Before deploying:**
1. Train model locally (creates model/ folder)
2. Commit model/ folder to git
3. Push to GitHub
4. Deploy

If model files are too large for git:
```bash
git lfs track "model/*.joblib"
```
