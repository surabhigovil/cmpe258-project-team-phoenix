# -*- coding: utf-8 -*-

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='/usr/local/airflow/scripts/weights_best_vgg16_model2.hdf5'

# MODEL_PATH = 'weights_best_vgg16_model2.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)





def model_predict(img_path, model):
    img_size = 160
    img = image.load_img(img_path, target_size=(img_size, img_size))
    #im = cv2.resize(cv2.cvtColor((img, cv2.COLOR_BGR2RGB), (img_size,img_size)).astype(np.float32)) / 255.0
    im = np.expand_dims(img, axis =0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    preds = model.predict(im)
    print("predictions ", preds, file=sys.stdout)
    sys.stdout.flush()
    preds=np.argmax(preds)
    if preds==0:
        preds="Safe driving"
    elif preds==1:
        preds="Texting on the phone - right side"
    elif preds==2:
        preds="Talking on the phone - right side"
    elif preds==3:
        preds="Texting on the phone - left side"
    elif preds==4:
        preds="Texting on the phone - left side"
    elif preds==5:
        preds="Adjusting the audio/console"
    elif preds==6:
        preds="Drink in hand"
    elif preds==7:
        preds="Reaching behind"
    elif preds==8:
        preds="Looking Away"
    elif preds==9:
        preds="Talking to fellow passenger"
    else:
        preds="Not recognized"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("in upload method")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, secure_filename(f.filename))
        f.save(file_path)
        print("saved file ")
        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    #Added hostname and port
    app.run(host = "0.0.0.0",port = 8008)
