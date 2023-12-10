from flask import Flask, render_template, request
import numpy as np
import pickle


import joblib

# initialise the app
app=Flask(__name__)

@app.route('/')
def form():
    # connecting html page from template folder to flask
    # By default folder name is  'templates' which contains files
    return render_template('diabetes.html')

@app.route('/predict',methods=['post'])
def predict():
    value1 = int(request.form.get('Pregnancies'))
    value2 = int(request.form.get('Glucose'))
    value3 = int(request.form.get('BloodPressure'))
    value4 = int(request.form.get('SkinThickness'))
    value5 = int(request.form.get('Insulin'))
    value6 = float(request.form.get('BMI'))
    value7 = float(request.form.get('DiabetesPedigreeFunction'))
    value8 = int(request.form.get('Age'))
    
    
    print("Value1=",value1,"value2=" ,value2)
    input_data = np.array([[value1, value2, value3, value4, value5, value6, value7, value8]])
    # Load the model
    #model = joblib.load('./save_model/model_save_80.pkl')
    model = joblib.load('model_save_80.pkl')


    # Make predictions
    prediction = model.predict(input_data)

    prediction1=int(prediction)
    print(prediction1)
    if prediction1 == 1:
        return "Person is diabitic"

    return "person is non diabitic"


    # print(prediction1)
    # return "prediction1"


# run the app
app.run(debug=True)