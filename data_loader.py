import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(symbol, start_date, end_date):
    """
    Fetches the stock data using yfinance.
    """
    return fetch_stock_data(symbol, start_date, end_date)

def fetch_stock_data(symbol, start_date, end_date):
    """
    Uses yfinance to download historical stock data.
    """
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}.")
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def compute_technical_indicators(data):
    """
    Computes additional technical indicators to enhance feature representation.
    """
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    
    # Fill NaN values
    data.bfill(inplace=True)

    return data

def preprocess_data(data, window_size=60, batch_size=32):
    """
    Preprocesses the stock data and returns DataLoaders with additional features.
    """
    data = compute_technical_indicators(data)
    
    # Select relevant features
    feature_columns = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'Volume']
    feature_data = data[feature_columns].values
    
    # Scale data to [0,1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_data)
    
    # Create sequences
    sequences = []
    targets = []
    for i in range(window_size, len(scaled_data)):
        sequences.append(scaled_data[i-window_size:i, :])  # Use all feature columns
        targets.append(scaled_data[i, 0])  # Target is 'Close' price
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Split into training and testing sets (80% train, 20% test)
    split_index = int(len(sequences) * 0.8)
    train_sequences = sequences[:split_index]
    train_targets = targets[:split_index]
    test_sequences = sequences[split_index:]
    test_targets = targets[split_index:]
    
    # Convert to PyTorch tensors
    train_sequences = torch.from_numpy(train_sequences).float()
    train_targets = torch.from_numpy(train_targets).float().unsqueeze(-1)
    test_sequences = torch.from_numpy(test_sequences).float()
    test_targets = torch.from_numpy(test_targets).float().unsqueeze(-1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_sequences, train_targets)
    test_dataset = TensorDataset(test_sequences, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler
