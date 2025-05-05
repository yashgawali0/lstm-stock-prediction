from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)
model = load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def get_data(stock):
    df = yf.download(stock, period="90d")
    data = df['Close'].values.reshape(-1, 1)
    scaled = scaler.transform(data)
    return scaled[-60:]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        stock = request.form['stock']
        data = get_data(stock)
        data = data.reshape(1, 60, 1)
        pred = model.predict(data)
        prediction = scaler.inverse_transform(pred)[0][0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
