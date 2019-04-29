#!/usr/bin/env python

from flask import Flask, request 
from keras.models import load_model
from keras import backend as K

import numpy 
app = Flask(__name__)

@app.route('/') 
def hello_world(): 
    return 'Index page'

@app.route('/predict', methods=['POST']) 
def add(): 
    req_data = request.get_json() 
    bizprop = req_data['bizprop'] 
    rooms = req_data['rooms'] 
    age = req_data['age'] 
    highways = req_data['highways'] 
    tax = req_data['tax'] 
    ptratio = req_data['ptratio'] 
    lstat = req_data['lstat'] 
    # This is where we load the actual saved model into new variable. 
    deep_and_wide_net = load_model('deep_and_wide_net.h5') 
    # Now we can use this to predict on new data 
    value = deep_and_wide_net.predict_on_batch(numpy.array([[bizprop, rooms, age  ,  highways   , tax   ,  ptratio  ,   lstat]], dtype=float)) 
    K.clear_session()

    return str(value)
