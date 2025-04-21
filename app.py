from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Load the trained model
model_path = 'model1.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer
vectorizer_path = 'vectorizer1.pkl'
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract text input
        email_text = request.form['email']

        # Transform using the same vectorizer used during training
        features = vectorizer.transform([email_text])

        # Predict
        prediction = model.predict(features)[0]

        # Map output to label
        class_names = {0: 'ham', 1: 'spam'}
        result = class_names.get(prediction, "Unknown")

        return render_template('index.html', prediction_text=f'Predicted email classification: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


# Optional health check route
@app.route('/ping')
def ping():
    return "App is running!", 200

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
