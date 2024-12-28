from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

## Route for homepage
@app.route('/')
def index():
    return render_template('home.html')

## Route for Single data point prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        ST1 = int(request.form.get("ST1"))
        ST2 = float(request.form.get('ST2'))
        PUT = float(request.form.get('PUT'))
        Attendance = float(request.form.get('Attendance'))

        new_data = scaler.transform([[ST1, ST2, PUT, Attendance]])
        predict = model.predict(new_data)

        result = f"Predicted Internal Marks: {predict[0]:.2f}"
        return render_template('result.html', result=result)

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
