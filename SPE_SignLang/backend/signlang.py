import os
import socket
import json
import logging
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

app = Flask(__name__)

def setup_logger():
    log_dir = '/var/log/app'
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('prediction-service')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(f'{log_dir}/backend.log', mode='a')  # Changed to backend.log
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_record = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
                "service": "prediction-service",
                "level": record.levelname.lower(),
                "message": record.getMessage(),
                "host": socket.gethostname()
            }
            return json.dumps(log_record)
    
    formatter = JSONFormatter()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logger configured successfully")
    return logger

logger = setup_logger()

def build_model():
    logger.info("Building model architecture...")
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.AveragePooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(26, activation='softmax')
    ])
    logger.info("Model architecture built successfully")
    return model

try:
    logger.info("Loading model weights...")
    model = build_model()
    model.load_weights("model_weights.h5")
    logger.info("Model weights loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model weights: {str(e)}", exc_info=True)
    raise

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")

    if 'file' not in request.files:
        logger.warning("No file uploaded in the request")
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files['file']
        logger.info(f"Processing file: {file.filename}")
        
        image = Image.open(file.stream).convert("RGB").resize((64, 64))
        img_array = np.array(image).reshape(1, 64, 64, 3) / 255.0

        logger.info("Making prediction...")
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_letter = chr(ord('A') + predicted_index)

        logger.info(f"Prediction successful - Letter: {predicted_letter}, Index: {predicted_index}")

        return jsonify({
            "prediction_index": predicted_index,
            "prediction_letter": predicted_letter
        })
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    logger.info("Starting sign language prediction service on port 5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.critical(f"Service failed to start: {str(e)}", exc_info=True)
        raise
