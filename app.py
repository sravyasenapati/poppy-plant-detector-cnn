from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust based on your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        return "Poppy Plant Detected 🌺"
    else:
        return "Not a Poppy Plant ❌"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    file_path = os.path.join("static", file.filename)
    file.save(file_path)

    result = predict_image(file_path)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)