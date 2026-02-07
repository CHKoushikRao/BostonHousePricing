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
"""
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    #print(data)
    data_array=np.array(list(data.values())).reshape(1,-1)
    #print(data_array)
    prediction=model.predict(data_array)
    #print(prediction)
    return jsonify(prediction[0])"""

@app.route('/predict',methods=['POST'])
def predict():
    data = {
        "longitude": float(request.form["longitude"]),
        "latitude": float(request.form["latitude"]),
        "housing_median_age": float(request.form["housing_median_age"]),
        "total_rooms": float(request.form["total_rooms"]),
        "total_bedrooms": float(request.form["total_bedrooms"]),
        "population": float(request.form["population"]),
        "households": float(request.form["households"]),
        "median_income": float(request.form["median_income"]),
        "ocean_proximity": request.form["ocean_proximity"]  # string stays string
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return render_template('home.html',prediction_text="The predicted price of the house is {}".format(prediction))


if __name__=="__main__":
    app.run(debug=True)