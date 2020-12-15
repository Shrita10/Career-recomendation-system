# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 09:30:57 2020

@author: Shrita
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model2 import LemmaTokenizer

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl','rb'))
le2 = pickle.load(open('le2.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model2 = pickle.load(open('model2.pkl','rb'))



@app.route('/')

def home():
    return render_template('untitled4.html')

@app.route('/predict',methods = ['POST'])

def predict():
    ''' For results on HTML GUI'''
    
    
    int_features = [float(x) for x in request.form.values() ]

    final_features = [np.array(int_features)]
    prediction = model1.predict(final_features)
    prediction = prediction*100
    output = round(prediction[0],2)
    
    return render_template('untitled4.html',prediction_text = 'According to the inputs, the likelihood of your university admission is {} %.'.format(output))

@app.route('/predict2',methods = ['POST'])
def predict2():
    ''' For results on HTML GUI'''
    
    input_text = request.form['resume']
    
    #input_text = [le2.transform([x]) if x.isalpha() else int(x) for x in request.form.values() ]
    vec_text = vectorizer.transform([input_text])
    output2 = model2.predict(vec_text)
    output2 = le2.inverse_transform([output2])
    output2 = output2[0]
    
    
    
    return render_template('untitled4.html',prediction_text2 = 'The suggested job title is {}, according to the qualifications entered.'.format(output2))

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    