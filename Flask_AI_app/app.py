from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist10_mobilenet.h5")
model = load_model(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from request
        data = request.json
        image_data = data['image']
        
        # Remove data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Open image with PIL
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Resize to 28x28 first (MNIST size)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize to 0-255 range
        img_array = 255 - img_array  # Invert colors (white background to black)
        
        # Convert grayscale to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
        
        # Resize to 96x96 for MobileNetV2
        img_array = tf.image.resize(np.expand_dims(img_array, 0), (96, 96))
        
        # Preprocess for MobileNet
        img_array = preprocess_input(img_array)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        
        # Get all probabilities
        all_predictions = [
            {'digit': i, 'probability': float(predictions[0][i])}
            for i in range(10)
        ]
        all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'success': True,
            'predicted_digit': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # For development
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For production with Gunicorn
    app.run(host='0.0.0.0', port=5000)