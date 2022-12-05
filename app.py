from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('Iris_flower.pkl','rb'))
std_scaler1 = pickle.load(open('object.obj','rb'))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('one.html')

@app.route('/predict', methods=['POST'])
def one():
    SepalLengthCm = request.form['SepalLengthCm']
    SepalWidthCm = request.form['SepalWidthCm']
    PetalLengthCm = request.form['PetalLengthCm']
    PetalWidthCm = request.form['PetalWidthCm']

    
    arr = np.array([[SepalLengthCm,SepalWidthCm, PetalLengthCm, PetalWidthCm,]]).reshape(1,-1)

    df1 = pd.DataFrame(arr)

    std_scaler2 = std_scaler1.transform(df1)
    pred = model.predict(std_scaler2)
    return render_template('two.html',data=pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)   