
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import io

app = Flask(__name__)

def build_model():
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
    return model

model = build_model()
model.load_weights("model_weights.h5")  

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB").resize((64, 64))
    img_array = np.array(image).reshape(1, 64, 64, 3) / 255.0

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_letter = chr(ord('A') + predicted_index)

    return jsonify({
        "prediction_index": predicted_index,
        "prediction_letter": predicted_letter
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
