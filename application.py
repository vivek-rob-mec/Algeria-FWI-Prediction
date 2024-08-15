from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application
## import ridge regressor model and Standard scale pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route("/") ## / for homepage
def index():
    return render_template('index.html') # render_template search for folder named 'templates'

@app.route("/predictdata", methods = ['GET','POST'])
def Predcit_DataPoint():        ## will be written same as url_for in home.html file
    if request.method =='POST':
        Temperature=float(request.form.get('Temperature'))  ## make sure the order is sames as table 
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled) # value stored in list format

        return render_template('home.html',result=result[0]) #result is shown on home page using result
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0") ## flask by default run on port 5000