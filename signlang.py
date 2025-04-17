# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from PIL import Image
import io
import uvicorn
from tensorflow.keras import models, layers, regularizers

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rebuild the model architecture
def build_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Conv2D(32, (5, 5), strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (5, 5), strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), strides=1, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(26, activation='softmax')  # 26 classes for A-Z
    ])

    return model

# Build and load weights
model = build_model()
model.load_weights("model_weights.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((64, 64))
    img_array = np.array(image).reshape(1, 64, 64, 3) / 255.0
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_char = chr(ord('A') + predicted_index)  # Maps 0–25 to A–Z
    return {
        "prediction_index": predicted_index,
        "prediction_letter": predicted_char
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

