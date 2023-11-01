from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from PIL import Image
import pandas as pd
import sklearn
import re
import random
from random import *

app = Flask(__name__)
model = load_model('skin1.h5')
model1 = load_model('DR.h5')
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/skin')
def diabetes():
    return render_template('skin.html')

def skin_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    return img



@app.route('/skin_result', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = skin_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)

    if prediction > 0.5:
        return render_template('skin.html',pred = "Skin Cancer is DETECTED. Please consult a Doctor!!!")
    else:
        return render_template('skin.html',pred = "Skin Cancer is NOT DETECTED. You are safe...")

@app.route('/dr')
def dr():
    return render_template('dr.html')

def dr_image(image):
    img = image.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    return img


@app.route('/dr_result', methods=['POST'])
def dr_result():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    img = dr_image(Image.open(image))
    img = np.expand_dims(img, axis=0)
    prediction = model1.predict(img)

    if prediction > 0.5:
        return render_template('dr.html',pred = "Diabetic Retinopathy is DETECTED. Please consult a Docror!!!")
    else:
        return render_template('dr.html',pred = "Diabetic Retinopathy is NOT DETECTED. You are safe...")


if __name__ == "__main__":
    app.run(debug=True)