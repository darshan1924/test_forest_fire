from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# import the ridge regression and standard scaler pickel file
ridge_model = pickle.load(open("models/ridge.pkl", 'rb'))
standard_scaler = pickle.load(open("models/scaler.pkl",'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    result = None

    if request.method == "POST":
        input_df = pd.DataFrame(
            [[
                float(request.form.get('temperature')),
                float(request.form.get('rh')),
                float(request.form.get('ws')),
                float(request.form.get('rain')),
                float(request.form.get('ffmc')),
                float(request.form.get('dmc')),
                float(request.form.get('isi')),
                float(request.form.get('classes')),
                float(request.form.get('region'))
            ]],
            columns=['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region']
        )

        scaled_data = standard_scaler.transform(input_df)
        result = ridge_model.predict(scaled_data)[0]

    return render_template("home.html", results=result)


    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
