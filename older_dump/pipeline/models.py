import torch
import torch.nn as nn
import numpy as np

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        out = self.fc(out)
        return self.sigmoid(out)

def create_sequences(data, target, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def save_gru_model(model, path):
    torch.save(model.state_dict(), path)

def load_gru_model(input_size, path):
    model = GRUModel(input_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
