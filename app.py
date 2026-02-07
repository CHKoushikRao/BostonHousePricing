import pickle 
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np
import pandas as pd

app=Flask(__name__)
##loading model
model=pickle.load(open('house_price_model.pkl','rb'))

@app.route ('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    #print(data)
    data_array=np.array(list(data.values())).reshape(1,-1)
    #print(data_array)
    prediction=model.predict(data_array)
    #print(prediction)
    return jsonify(prediction[0])

if __name__=="__main__":
    app.run(debug=True)