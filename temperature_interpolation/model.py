import torch
from torch import nn

"""Two-layer bidirectional LSTM for sequence-to-sequence reconstruction."""

class TemperatureBiLSTM(nn.Module):
    """Use both sides of each gap and return one temperature per input step."""

    def __init__(self, input_size, hidden_sizes=(64, 32), dropout=0.2, dense_size=16):
        super().__init__()
        self.first_lstm = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.second_lstm = nn.LSTM(2 * hidden_sizes[0], hidden_sizes[1], batch_first=True, bidirectional=True)
        self.output = nn.Sequential(nn.Linear(2 * hidden_sizes[1], dense_size), nn.ReLU(), nn.Linear(dense_size, 1))

    def forward(self, inputs):
        """Preserve batch and sequence axes; squeeze only the output channel."""
        first_sequence, _ = self.first_lstm(inputs)
        second_sequence, _ = self.second_lstm(self.dropout(first_sequence))
        return self.output(second_sequence).squeeze(-1)


def build_model(config, input_size):
    """Construct the configured architecture for either scenario."""
    return TemperatureBiLSTM(input_size, config.hidden_sizes, config.dropout, config.dense_size)


def masked_mse(predictions, truth, loss_mask):
    """Average squared error exclusively over artificially hidden positions."""
    if predictions.shape != truth.shape or truth.shape != loss_mask.shape:
        raise ValueError("Prediction, truth, and loss mask shapes must match.")
    selected = loss_mask.bool()
    if not torch.any(selected):
        raise ValueError("Loss requires at least one artificially hidden value.")
    return (predictions[selected] - truth[selected]).square().mean()
