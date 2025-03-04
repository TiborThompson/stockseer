# Stock Price Prediction Using LSTM

## Project Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical data. By leveraging patterns in time series data, the model aims to provide accurate forecasts of stock price movements. The project is organized into modular components, making it easier to understand, maintain, and extend.

## File Description

- **main.py**: The main script that orchestrates the entire workflow. It handles data acquisition, preprocessing, model training, evaluation, and making predictions using the LSTM model.

- **data_loader.py**: Responsible for fetching and preprocessing the stock price data. This module prepares the data for input into the LSTM model by handling tasks such as data normalization and splitting the data into training and testing sets.

- **model.py**: Defines the architecture of the LSTM neural network. It sets up the layers of the model and compiles it, making it ready for training on the preprocessed data.

- **utils.py**: Provides utility functions for model evaluation, calculating performance metrics (such as Mean Squared Error), and visualizing results. This includes plotting graphs of the predicted versus actual stock prices.

- **requirements.txt**: Lists all the Python package dependencies required to run the project. This ensures that anyone setting up the project will have all the necessary libraries installed.

## How to Run the Project

### Prerequisites

- Python 3.x installed on your system.
- An internet connection to fetch stock price data (if applicable).
- Virtual environment tool (optional but recommended), such as `virtualenv` or `venv`.

### Installation

1. **Clone the repository**: Download or clone the project repository to your local machine using your preferred method.

2. **Navigate to the project directory**: Open a terminal or command prompt and change the current directory to the project's root directory.

3. **Create a virtual environment** (optional but recommended):

   - **Install virtualenv** (if not already installed):

     ```
     pip install virtualenv
     ```

   - **Create a new virtual environment**:

     ```
     virtualenv venv
     ```

   - **Activate the virtual environment**:

     - On Windows:

       ```
       venv\Scripts\activate
       ```

     - On macOS/Linux:

       ```
       source venv/bin/activate
       ```

4. **Install the required packages**:

   ```
   pip install -r requirements.txt
   ```

### Usage

1. **Configure the data source**:

   - Open `data_loader.py` and set the parameters for data fetching.
     - Specify the stock ticker symbol you want to predict.
     - Set the date range for the historical data.
     - Ensure any API keys or authentication (if required) for data sources are correctly configured.

2. **Run the main script**:

   ```
   python main.py
   ```

   This script will perform the following steps:

   - Fetch and preprocess the stock price data.
   - Train the LSTM model using the training dataset.
   - Evaluate the model's performance on the testing dataset.
   - Make predictions on future stock prices.
   - Visualize the results by plotting graphs.

3. **View the outputs**:

   - **Console Outputs**: The script will display training progress and evaluation metrics in the terminal or command prompt.
   - **Plots**: Graphs comparing the actual and predicted stock prices will be displayed. These may also be saved to the project directory.

### Customization

- **Adjust Model Parameters**:

  - Open `model.py` to modify the LSTM architecture.
    - Change the number of layers, neurons, activation functions, etc.
  - Adjust training parameters in `main.py` or where the model is compiled.
    - Modify the number of epochs, batch size, learning rate, etc.

- **Modify Data Preprocessing**:

  - In `data_loader.py`, alter how data is scaled or normalized.
  - Change the way the dataset is split into training and testing sets.
  - Implement additional data augmentation or feature engineering techniques.

- **Enhance Utility Functions**:

  - Add new evaluation metrics in `utils.py`, such as Mean Absolute Error or R-squared.
  - Customize the plots for better visualization or to include additional information.

- **Extend to Multiple Stocks**:

  - Modify the scripts to handle multiple stock symbols simultaneously.
  - Aggregate and compare predictions across different stocks.

## Performance Metrics

When trained on Apple (AAPL) stock data with default parameters, the model achieves impressive performance metrics:

- **Mean Absolute Error (MAE)**: ~1.72% 
- **Mean Squared Error (MSE)**: ~4.87%
- **Root Mean Squared Error (RMSE)**: ~2.21%
- **Maximum Deviation**: ~10.90

These metrics indicate the model's ability to accurately predict stock price movements within a reasonable margin of error.

## Future Enhancements

Potential improvements for future versions:
- Add R² (coefficient of determination) as a performance metric
- Implement additional technical indicators and features
- Fine-tune hyperparameters for better performance
- Support for ensemble models
- Add backtesting capabilities with trading strategies

## Questions and Support

For any questions, issues, or suggestions regarding this project, please feel free to reach out through the project's repository or contact the maintainer directly.

---

**Note**: Always ensure you comply with the terms of service of any data providers you use, especially when fetching financial data. This project is for educational purposes and should not be used as the sole basis for any financial decisions.