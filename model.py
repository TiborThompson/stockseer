import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for stock price prediction.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        output_size (int): Number of outputs.
        dropout_prob (float): Dropout probability.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.0):
        super(LSTMModel, self).__init__()
        # Initialize the LSTM layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Initialize Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        # Initialize fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: Output predictions of shape (batch_size, output_size).
        """
        # LSTM computation
        out, (hn, cn) = self.lstm(x)
        # Take the output from the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        # Apply dropout
        out = self.dropout(out)
        # Pass through fully connected layer
        out = self.fc(out)
        return out
