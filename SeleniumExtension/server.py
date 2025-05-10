from flask import Flask, request, jsonify
from flask_cors import CORS
from cookie_extractor import CookieExtractor
import json
import os

app = Flask(__name__)
CORS(app)  # Cho phép mọi nguồn truy cập (CORS)
extractor = None

@app.route('/extract_cookies', methods=['POST'])
def extract_cookies():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    extractor = CookieExtractor(headless=True)
    try:
        cookies = extractor.extract_cookies(url)
        return jsonify({'cookies': cookies})
    finally:
        extractor.close()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sequence = data.get('sequence')
        
        if not sequence:
            return jsonify({'error': 'Sequence is required'}), 400
            
        # TODO: Add your prediction logic here
        # This is a placeholder response
        prediction = {
            'predicted_class': 'high',
            'probabilities': [0.1, 0.2, 0.3, 0.35, 0.05]
        }
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    global extractor
    try:
        if extractor:
            extractor.close()
            extractor = None
        return jsonify({'message': 'Cleanup successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 