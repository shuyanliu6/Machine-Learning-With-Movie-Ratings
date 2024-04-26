from flask import Flask, request, jsonify
import joblib
from textblob import TextBlob
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load trained pipeline 
pipeline = joblib.load('rf_pipeline.pkl')

app = Flask(__name__)
# Define Feature Engineering Functions
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def count_special_chars(text):
    special_chars = set("!@#$%^&*()_+-=[]{}|;:',.<>?")
    return sum(1 for char in text if char in special_chars)

# Define Endpoints for Prediction and Feedback

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email_text = data.get('email')
    if not email_text:
        return jsonify({'error': 'Missing email field'}), 400
    
    try:
        input_df = pd.DataFrame([{
            'cleaned_text': email_text,
            'sentiment_polarity': get_sentiment(email_text),
            'special_chars': count_special_chars(email_text)
        }])
        prediction = pipeline.predict(input_df)[0]  # Assume model returns a numpy array
        if prediction == 1:
            result = 'spam'
        else:
            result = 'ham'
        return jsonify({'result': result})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the request'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(force=True)
    if not data or 'email' not in data or 'is_spam' not in data:
        return jsonify({'error': 'Missing data fields'}), 400
    logging.info(f"Feedback received: {data}")
    return jsonify({'success': True})

# Model Deploy

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
