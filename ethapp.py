from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import warnings
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set Matplotlib to non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)

# Load Pre-trained Multi-Crypto Model and Scalers
try:
    model = load_model("model.keras")
    with open('crypto_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    print("Multi-cryptocurrency model and scalers loaded successfully!")
    print(f"Model uses cryptocurrencies: BTC-USD, ETH-USD, DOGE-USD")
    print("Model predicts BTC-USD prices using all three cryptocurrencies as features.")
except Exception as e:
    print(f"Error loading model or scalers: {e}")
    model = None
    scalers = None

# Helper Function to Convert Matplotlib Plots to HTML
def plot_to_html(fig):
    """
    Convert a matplotlib figure to base64 HTML format for embedding in web pages
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        crypto_choice = request.form.get("crypto_choice")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", crypto_choice=crypto_choice, no_of_days=no_of_days))
    return render_template("index.html", title="Cryptocurrency Price Predictor")

@app.route("/predict")
def predict():
    try:
        # Check if model and scalers are loaded
        if model is None or scalers is None:
            return render_template("result.html", 
                                 title="Cryptocurrency Price Predictor - Error",
                                 error="Model or scalers not loaded. Please check if model.keras and crypto_scalers.pkl files exist.")
            
        crypto_choice = request.args.get("crypto_choice", "BTC-USD")
        no_of_days = int(request.args.get("no_of_days", 10))

        # Validate input parameters
        if no_of_days < 1 or no_of_days > 100:
            return render_template("result.html", 
                                 title="Cryptocurrency Price Predictor - Error",
                                 error="Number of days must be between 1 and 100.")

        # Fetch data for all three cryptocurrencies (needed for the model)
        stocks = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
        end = datetime.now()
        start = datetime(end.year - 10, end.month, end.day)
        
        crypto_data = {}
        for stock in stocks:
            crypto_data[stock] = yf.download(stock, start, end)
            if crypto_data[stock].empty:
                return render_template("result.html", 
                                     title="Cryptocurrency Price Predictor - Error",
                                     error=f"No data available for {stock}.")
        
        # Create combined closing prices dataframe
        closing_prices = pd.DataFrame()
        for stock in stocks:
            closing_prices[stock] = crypto_data[stock]['Close']
        
        # Remove NaN values
        clean_closing_prices = closing_prices.dropna()
        
        if len(clean_closing_prices) < 200:
            return render_template("result.html", 
                                 title="Cryptocurrency Price Predictor - Error",
                                 error="Not enough historical data for prediction (minimum 200 days required).")

        # Scale data using the saved scalers
        scaled_data_dict = {}
        for stock in stocks:
            scaled_data_dict[stock] = scalers[stock].transform(clean_closing_prices[[stock]])

        # Combine all scaled data into a single array for multi-feature input
        scaled_data_combined = np.column_stack([scaled_data_dict[stock].flatten() for stock in stocks])

        # Prepare test data (last 90% for consistency with training)
        splitting_len = int(len(scaled_data_combined) * 0.1)  # Use last 10% as test
        test_data = scaled_data_combined[splitting_len:]
        
        if len(test_data) < 100:
            return render_template("result.html", 
                                 title="Cryptocurrency Price Predictor - Error",
                                 error="Not enough test data for prediction.")

        # Prepare data for model prediction
        x_data = []
        y_data = []
        base_days = 100
        
        # Determine which crypto to predict based on user choice
        crypto_index = {'BTC-USD': 0, 'ETH-USD': 1, 'DOGE-USD': 2}[crypto_choice]
        
        for i in range(base_days, len(test_data)):
            x_data.append(test_data[i-base_days:i])
            y_data.append(test_data[i, crypto_index])  # Use selected crypto index

        if len(x_data) == 0:
            return render_template("result.html", 
                                 title="Cryptocurrency Price Predictor - Error",
                                 error="Insufficient data for model prediction.")

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Make predictions
        predictions = model.predict(x_data)
        inv_predictions = scalers[crypto_choice].inverse_transform(predictions.reshape(-1, 1))
        inv_y_test = scalers[crypto_choice].inverse_transform(y_data.reshape(-1, 1))

        # Create index for plotting
        start_idx = splitting_len + base_days
        end_idx = start_idx + len(inv_y_test)
        plot_index = clean_closing_prices.index[start_idx:end_idx]

        # Prepare Data for Plotting
        plotting_data = pd.DataFrame({
            'Original Test Data': inv_y_test.flatten(),
            'Predicted Test Data': inv_predictions.flatten()
        }, index=plot_index)

        # Generate Plots
        # Plot 1: Selected cryptocurrency
        fig1 = plt.figure(figsize=(15, 6))
        plt.plot(clean_closing_prices.index, clean_closing_prices[crypto_choice], 
                 label=f'{crypto_choice} Close Price', color='blue', linewidth=2)
        plt.title(f"{crypto_choice} Price History")
        plt.xlabel("Date")
        plt.ylabel("Close Price (USD)")
        plt.legend()
        plt.grid(alpha=0.3)
        original_plot = plot_to_html(fig1)
        plt.close(fig1)

        # Plot 2: Original vs Predicted Test Data
        fig2 = plt.figure(figsize=(15, 6))
        plt.plot(plotting_data['Original Test Data'], label=f"Original {crypto_choice} Test Data", color='blue')
        plt.plot(plotting_data['Predicted Test Data'], label=f"Predicted {crypto_choice} Test Data", linestyle="--", color='red')
        plt.legend()
        plt.title(f"Original vs Predicted {crypto_choice} Closing Prices")
        plt.xlabel("Date")
        plt.ylabel(f"{crypto_choice} Close Price (USD)")
        plt.grid(alpha=0.3)
        predicted_plot = plot_to_html(fig2)
        plt.close(fig2)

        # Plot 3: Future Predictions
        last_100 = scaled_data_combined[-100:].reshape(1, 100, 3)  # 3 features
        future_predictions = []

        for day in range(no_of_days):
            next_day_prediction = model.predict(last_100, verbose=0)
            future_predictions.append(scalers[crypto_choice].inverse_transform(next_day_prediction.reshape(-1, 1)))
            
            # Update input for next prediction - update all three cryptos
            new_input = np.zeros((1, 1, 3))
            # For simplicity, we'll update the predicted crypto and keep others constant
            if crypto_choice == 'BTC-USD':
                new_input[0, 0, 0] = next_day_prediction[0, 0]  # Predicted BTC
                new_input[0, 0, 1] = last_100[0, -1, 1]  # Last ETH value
                new_input[0, 0, 2] = last_100[0, -1, 2]  # Last DOGE value
            elif crypto_choice == 'ETH-USD':
                new_input[0, 0, 0] = last_100[0, -1, 0]  # Last BTC value
                new_input[0, 0, 1] = next_day_prediction[0, 0]  # Predicted ETH
                new_input[0, 0, 2] = last_100[0, -1, 2]  # Last DOGE value
            else:  # DOGE-USD
                new_input[0, 0, 0] = last_100[0, -1, 0]  # Last BTC value
                new_input[0, 0, 1] = last_100[0, -1, 1]  # Last ETH value
                new_input[0, 0, 2] = next_day_prediction[0, 0]  # Predicted DOGE
            
            last_100 = np.append(last_100[:, 1:, :], new_input, axis=1)

        future_predictions = np.array(future_predictions).flatten()

        fig3 = plt.figure(figsize=(15, 6))
        plt.plot(range(1, no_of_days + 1), future_predictions, marker='o', 
                label=f"Predicted Future {crypto_choice} Prices", color="purple", linewidth=2)
        plt.title(f"Future {crypto_choice} Price Predictions")
        plt.xlabel("Days Ahead")
        plt.ylabel(f"Predicted {crypto_choice} Close Price (USD)")
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Add price labels on points
        for i, val in enumerate(future_predictions):
            plt.text(i+1, val, f'${val:.2f}', fontsize=10, ha='center', va='bottom', color='black')
        
        future_plot = plot_to_html(fig3)
        plt.close(fig3)

        return render_template(
            "result.html",
            title="Cryptocurrency Price Predictor - Results",
            stock=crypto_choice,
            original_plot=original_plot,
            predicted_plot=predicted_plot,
            future_plot=future_plot,
            future_predictions=future_predictions
        )
        
    except Exception as e:
        return render_template("result.html", 
                             title="Cryptocurrency Price Predictor - Error", 
                             error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)