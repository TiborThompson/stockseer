import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data_loader import load_data, preprocess_data
from model import LSTMModel
from utils import plot_predictions, calculate_metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Stock Price Prediction using LSTM')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--start_date', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-10-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--window_size', type=int, default=60, help='Window size for input sequences')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='Number of hidden units in LSTM')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Fetch and preprocess data
    data = load_data(args.symbol, args.start_date, args.end_date)
    train_loader, test_loader, scaler = preprocess_data(data, args.window_size, args.batch_size)

    # Initialize the model
    # Initialize the model
    input_size = 7  # Number of features (Close, SMA_20, SMA_50, EMA_20, RSI, MACD, Volume)
    hidden_size = args.hidden_size 
    num_layers = 1 
    output_size = 1 

    model = LSTMModel(input_size, hidden_size, num_layers, output_size) 


    # Define optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}')

    # Evaluation
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())
            actuals.append(targets.cpu())

    # Concatenate predictions and actuals
    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()
    # Reverse scaling
    # Expand predictions to match the original feature shape
    predictions_expanded = np.zeros((predictions.shape[0], 7))  # 7 features used in training
    predictions_expanded[:, 0] = predictions[:, 0]  # Only fill 'Close' column

    # Apply inverse transform
    predictions = scaler.inverse_transform(predictions_expanded)[:, 0]  # Extract only the 'Close' values


    # Expand actuals to match the original feature shape (7 features)
    actuals_expanded = np.zeros((actuals.shape[0], 7))  # Create an empty array with 7 columns
    actuals_expanded[:, 0] = actuals[:, 0]  # Fill only the 'Close' column

    # Apply inverse transform
    actuals = scaler.inverse_transform(actuals_expanded)[:, 0]  # Extract only 'Close' values


    # Calculate performance metrics
    metrics = calculate_metrics(actuals, predictions)
    if metrics is None:
        print("Error: Metrics calculation failed.")
        return

    print(f"MAE: {metrics['MAE (%)']:.2f}%, MSE: {metrics['MSE (%)']:.2f}%, RMSE: {metrics['RMSE (%)']:.2f}%, "
        f"Max Deviation: {metrics['Max Deviation']:.4f}, Standard MSE: {metrics['Standard MSE']:.4f}")




    # Plot actual vs predicted
    plot_predictions(actuals, predictions, args.symbol)

    # Save the model
    torch.save(model.state_dict(), f'{args.symbol}_lstm_model.pth')

if __name__ == "__main__":
    main()