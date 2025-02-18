import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(predictions, targets):
    """
    Computes evaluation metrics between predictions and targets in percentage terms.

    Args:
        predictions (np.ndarray): Predicted values.
        targets (np.ndarray): Actual values.

    Returns:
        dict: Dictionary containing percentage-based MAE, MSE, RMSE, maximum deviation, and standard MSE.
    """
    try:
        # Convert to numpy arrays and flatten
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()

        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        # Avoid division by zero
        if np.any(targets == 0):
            raise ValueError("Targets contain zero values, cannot compute percentage error.")

        percentage_errors = np.abs((predictions - targets) / targets) * 100
        mae = np.mean(percentage_errors)
        mse = np.mean(percentage_errors ** 2)
        rmse = np.sqrt(mse)

        # Additional Metrics
        max_deviation = np.max(np.abs(predictions - targets))  # Maximum absolute deviation
        standard_mse = mean_squared_error(targets, predictions)  # Standard MSE

        return {'MAE (%)': mae, 'MSE (%)': mse, 'RMSE (%)': rmse, 'Max Deviation': max_deviation, 'Standard MSE': standard_mse}
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        return None

def plot_predictions(actual_prices, predicted_prices, title):
    """
    Plots actual vs. predicted stock prices and saves the plot.

    Args:
        actual_prices (np.ndarray): Actual stock prices.
        predicted_prices (np.ndarray): Predicted stock prices.
        title (str): Title for the plot.
    """
    try:
        actual_prices = np.array(actual_prices).flatten()
        predicted_prices = np.array(predicted_prices).flatten()

        if actual_prices.shape != predicted_prices.shape:
            raise ValueError("Actual prices and predicted prices must have the same shape.")

        plt.figure(figsize=(12, 6))
        plt.plot(actual_prices, label='Actual Prices')
        plt.plot(predicted_prices, label='Predicted Prices')
        plt.title(f'Actual vs Predicted Stock Prices for {title}')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.tight_layout()

        # Save the plot to disk
        plt.savefig(f'{title}_predictions.png')

        # Optionally display the plot
        plt.show()
    except Exception as e:
        print(f"Error in plot_predictions: {e}")

def plot_training_loss(loss_values):
    """
    Plots the training loss over epochs and saves the plot.

    Args:
        loss_values (list or np.ndarray): List of loss values over epochs.
    """
    try:
        if not isinstance(loss_values, (list, np.ndarray)):
            raise ValueError("Loss values must be a list or numpy array.")

        plt.figure(figsize=(8, 6))
        plt.plot(loss_values, label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        # Save the plot to disk
        plt.savefig('training_loss.png')

        # Optionally display the plot
        plt.show()
    except Exception as e:
        print(f"Error in plot_training_loss: {e}")

def inverse_transform(scaler, data):
    """
    Inverses the scaling transformation applied to the data.

    Args:
        scaler (sklearn.preprocessing scaler): Fitted scaler object.
        data (np.ndarray): Scaled data to invert.

    Returns:
        np.ndarray: Original data before scaling.
    """
    try:
        return scaler.inverse_transform(data)
    except Exception as e:
        print(f"Error in inverse_transform: {e}")
        return None
