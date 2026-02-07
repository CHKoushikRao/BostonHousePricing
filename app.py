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

@app.route('/predict', methods=['POST'])
def predict():
    form_data = {
        "longitude": request.form["longitude"],
        "latitude": request.form["latitude"],
        "housing_median_age": request.form["housing_median_age"],
        "total_rooms": request.form["total_rooms"],
        "total_bedrooms": request.form["total_bedrooms"],
        "population": request.form["population"],
        "households": request.form["households"],
        "median_income": request.form["median_income"],
        "ocean_proximity": request.form["ocean_proximity"]
    }

    df = pd.DataFrame([{
        "longitude": float(form_data["longitude"]),
        "latitude": float(form_data["latitude"]),
        "housing_median_age": float(form_data["housing_median_age"]),
        "total_rooms": float(form_data["total_rooms"]),
        "total_bedrooms": float(form_data["total_bedrooms"]),
        "population": float(form_data["population"]),
        "households": float(form_data["households"]),
        "median_income": float(form_data["median_income"]),
        "ocean_proximity": form_data["ocean_proximity"]
    }])

    prediction = model.predict(df)[0]

    return render_template(
        "home.html",
        prediction_text=f"The predicted house price is {prediction}",
        form_data=form_data
    )

if __name__=="__main__":
    app.run(debug=True)