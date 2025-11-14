// Configuration
const API_BASE_URL = 'https://ai-vs-human-text-classifier-wjl3.onrender.com';
const API_ENDPOINTS = {
    predict: '/predict',
    health: '/health',
    root: '/'
};

// Example texts
const EXAMPLES = {
    1: "Machine learning, a subset of artificial intelligence, facilitates the optimization of computational processes through advanced iterative refinement methodologies and algorithmic implementations.",
    2: "hey whats up lol just grabbed some coffee brb gonna finish this later",
    3: "The implementation of advanced neural network architectures enables the enhancement of performance metrics across diverse application domains while maintaining computational efficiency.",
    4: "I honestly think this whole situation is ridiculous. Nobody asked for this change and now we're all stuck dealing with the consequences. It's frustrating when decisions are made without considering the people actually affected."
};

// Check API health on load
window.addEventListener('load', checkAPIHealth);

// Check API Health
async function checkAPIHealth() {
    const statusElement = document.getElementById('apiStatus');

    try {
        const response = await fetch(API_BASE_URL + API_ENDPOINTS.health);
        const data = await response.json();

        if (response.ok && data.model_loaded) {
            statusElement.textContent = '✅ API Online & Ready';
            statusElement.parentElement.className = 'api-status online';
        } else {
            statusElement.textContent = '⚠️ API Online (Model Issue)';
            statusElement.parentElement.className = 'api-status offline';
        }
    } catch (error) {
        statusElement.textContent = '❌ API Offline';
        statusElement.parentElement.className = 'api-status offline';
        console.error('Health check failed:', error);
    }
}

// Load example text
function loadExample(num) {
    document.getElementById('textInput').value = EXAMPLES[num];
}

// Analyze text
async function analyzeText() {
    const text = document.getElementById('textInput').value.trim();
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const loadingDiv = document.getElementById('loading');
    const analyzeBtn = document.querySelector('.analyze-btn');

    // Reset UI
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';

    // Validate input
    if (!text) {
        showError('Please enter some text to analyze!');
        return;
    }

    // Show loading state
    loadingDiv.style.display = 'block';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch(API_BASE_URL + API_ENDPOINTS.predict, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        displayResult(data);

    } catch (error) {
        console.error('Error:', error);
        showError('Failed to analyze text. Please check if the API is running and try again.');
    } finally {
        loadingDiv.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display result
function displayResult(data) {
    const resultDiv = document.getElementById('result');

    // Set label with emoji
    const emoji = data.label === 'AI' ? '🤖' : '✍️';
    const labelText = data.label === 'AI' ? 'AI Generated' : 'Human Written';
    document.getElementById('resultLabel').innerHTML = `${emoji} ${labelText}`;

    // Set probabilities
    document.getElementById('aiProb').textContent = 
        (data.ai_probability * 100).toFixed(1) + '%';
    document.getElementById('humanProb').textContent = 
        (data.human_probability * 100).toFixed(1) + '%';

    // Set confidence
    const confidenceMap = {
        'very_high': '🟢 Very High',
        'high': '🟡 High',
        'medium': '🟠 Medium',
        'low': '🔴 Low'
    };
    document.getElementById('confidence').textContent = 
        confidenceMap[data.confidence] || data.confidence.toUpperCase();

    // Show result with appropriate styling
    resultDiv.className = 'result ' + data.label.toLowerCase();
    resultDiv.style.display = 'block';
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = '❌ ' + message;
    errorDiv.style.display = 'block';
}

// Clear all
function clearAll() {
    document.getElementById('textInput').value = '';
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display = 'none';
}

// Keyboard shortcut: Ctrl/Cmd + Enter to analyze
document.getElementById('textInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        analyzeText();
    }
});

// Debug: Log API configuration
console.log('AI Text Detector Initialized');
console.log('API Base URL:', API_BASE_URL);
console.log('Endpoints:', API_ENDPOINTS);
