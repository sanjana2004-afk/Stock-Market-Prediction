import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify

# Initialize Flask app
app = Flask(__name__)
plt.style.use("fivethirtyeight")

# Load pre-trained model
model = load_model('model.h5')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock = request.form.get('stock')
        if not stock:
            return render_template('predict.html', error="Stock ticker is required.")

        # Fetch data from Yahoo Finance
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()
        df = yf.download(stock, start=start, end=end)
        
        if df.empty:
            return render_template('predict.html', error="No data found for this ticker.")

        # Calculate EMAs
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Split data
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], axis=0)
        input_data = scaler.transform(final_df)

        # Prepare test data
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Make predictions
        y_predicted = model.predict(x_test)
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot and save charts
        if not os.path.exists('static'):
            os.makedirs('static')

        # EMA 20 & 50
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.set_title("Closing Price vs Time (20 & 50 Days EMA)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = os.path.join("static", "ema_20_50.png")
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # EMA 100 & 200
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.set_title("Closing Price vs Time (100 & 200 Days EMA)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = os.path.join("static", "ema_100_200.png")
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Prediction vs Actual
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Actual Price", linewidth=1)
        ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = os.path.join("static", "stock_prediction.png")
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Save dataset as CSV
        csv_file_path = os.path.join("static", f"{stock}_dataset.csv")
        df.to_csv(csv_file_path)

        # Render results in predict page
        return render_template('predict.html',
                               plot_path_ema_20_50=ema_chart_path,
                               plot_path_ema_100_200=ema_chart_path_100_200,
                               plot_path_prediction=prediction_chart_path,
                               data_desc=df.describe().to_html(classes='table table-bordered'),
                               dataset_link=f"{stock}_dataset.csv")

    except Exception as e:
        return render_template('predict.html', error=str(e))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join("static", filename), as_attachment=True)

# ===== MAIN =====
if __name__ == '__main__':
    app.run(debug=True)
