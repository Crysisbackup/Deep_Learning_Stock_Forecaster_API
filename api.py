# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
from datetime import datetime
from tensorflow import keras
import yfinance as yf
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the input data schema using Pydantic
class InputData(BaseModel):
    start: str
    end: str
    stock: str

# Initialize FastAPI app
app = FastAPI(title="Stock Price Prediction API")

# Load the model during startup
try:
    model_path = os.path.join("Stock Predictions Model.keras")
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        model = None
    else:   
        model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
    scaler_path = os.path.join('scaler.pkl')
    if not os.path.exists(scaler_path):
            logger.warning(f"Scaler file not found: {scaler_path}")
            scaler = None
    else:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    model = None
    scaler = None




@app.get("/")
def root():
    """Root endpoint to verify API is running"""
    return {
        "message": "Stock Prediction API is running",
        "status": "online",
        "endpoints": {
            "POST /predict": "Make stock predictions",
            # "GET /health": "Check API health",
            # "GET /docs": "API documentation"
        }
    }  

@app.post("/predict")
def predict(data: InputData):
    # Convert datetimes to 
    try:
         
        start_num = datetime.strptime(data.start, "%Y-%m-%d")
        end_num = datetime.strptime(data.end, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid Date format: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}")

    # Download stock data
    stock_data = yf.download(data.stock, start=data.start, end=data.end)
    if stock_data.empty:
        raise HTTPException(status_code= 404, detail="No data found for the given stock and date range.")
    
#     with open('scaler.pkl', 'rb') as f:
#         scaler = pickle.load(f)

    close_prices = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)

    sequence_length = 100
    if len(scaled_data) < sequence_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Need at least {sequence_length} days of data. Got {len(scaled_data)} days."
            )
    
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    
    
    
     

    # Make prediction using the loaded model
    prediction = model.predict(last_sequence)
    # ✅ Correct approach:
    scaled_prediction = model.predict(last_sequence)  # Gets 0.73
    actual_price = scaler.inverse_transform(scaled_prediction)  # Converts 0.73 → $156.20
    return {
         "predicted_stock_price": float(actual_price[0][0]),
         "stock_symbol": data.stock,
         "prediction_date": end_num.strftime("%Y-%m-%d")}
 

   

