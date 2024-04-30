# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:09:10 2024

@author: Samson
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
HR_Recruitment = pickle.load(open('HR_Recruitment.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = HR_Recruitment.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Starting Salary Forcast $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = HR_Recruitment.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)