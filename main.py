import os
import uvicorn
import traceback
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO

import numpy as np

from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array

model = tf.keras.models.load_model('EfficientNetB0_classifier.h5')

labels = ['aloevera',
          'banana',
          'bilimbi',
          'cantaloupe',
          'cassava',
          'coconut',
          'corn',
          'cucumber',
          'curcuma',
          'eggplant',
          'galangal',
          'ginger',
          'guava',
          'kale',
          'longbeans',
          'mango',
          'melon',
          'orange',
          'paddy',
          'papaya',
          'peper chili',
          'pineapple',
          'pomelo',
          'shallot',
          'soybeans',
          'spinach',
          'sweet potatoes',
          'tobacco',
          'waterapple',
          'watermelon']

app = FastAPI()

@app.get("/")
def index():
    return "Hello world from ML endpoint!"

@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        # Checking if it's an image
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        img = load_image_into_numpy_array(uploaded_file.file.read())

        results = model.predict(img)
        results = results[0]
        likely_class = np.argmax(results)
        confidence_score = float(results[likely_class])

        response_data = {"predicted_class": labels[int(likely_class)], "confidence_score": confidence_score}

        return response_data
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


# Starting the server
# Your can check the API documentation easily using /docs after the server is running
if __name__ == "__main__":
    # Starting the server
    # Your can check the API documentation easily using /docs after the server is running
    port = os.environ.get("PORT", 8080)
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port)