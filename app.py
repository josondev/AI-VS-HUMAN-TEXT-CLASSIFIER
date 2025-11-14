# Flask API for AI Text Detector - Deploy to Render/Railway

from flask import Flask, request, jsonify
from flask_cors import CORS
from detector import AITextDetector
import os

app = Flask(__name__)
CORS(app)

# Load model on startup
detector = AITextDetector()
MODEL_DIR = 'model'

try:
    if detector.model_exists(MODEL_DIR):
        detector.load_model(MODEL_DIR)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  No model found - train first!")
except Exception as e:
    print(f"❌ Error loading model: {e}")


@app.route('/')
def home():
    """API info."""
    return jsonify({
        'status': 'online',
        'name': 'AI Text Detector API',
        'model_loaded': detector.is_trained,
        'endpoints': {
            'POST /predict': 'Classify text',
            'GET /health': 'Health check'
        }
    })


@app.route('/health')
def health():
    """Health check."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.is_trained
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Classify text as AI or Human."""
    try:
        if not detector.is_trained:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        text = data['text'].strip()

        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Predict
        label, ai_prob, human_prob = detector.predict(text)

        # Confidence
        certainty = abs(ai_prob - human_prob)
        if certainty > 0.8:
            confidence = "very_high"
        elif certainty > 0.5:
            confidence = "high"
        elif certainty > 0.2:
            confidence = "medium"
        else:
            confidence = "low"

        return jsonify({
            'label': label,
            'ai_probability': round(ai_prob, 4),
            'human_probability': round(human_prob, 4),
            'confidence': confidence,
            'certainty': round(certainty, 4),
            'text_length': len(text),
            'word_count': len(text.split())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
