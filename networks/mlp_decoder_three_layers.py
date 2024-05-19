import torch
import torch.nn.functional as F
from torch import nn


class ThreeLayersDecoder(nn.Module):
    def __init__(self, input_size, output_size, fc1_size=500, fc2_size=500, dropout_ratio=0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.fc3 = nn.Linear(fc2_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def predict_proba(self, x):
        out = self.forward(x)
        if self.output_size == 1:
            return torch.sigmoid(out).reshape(-1)
        else:
            return torch.softmax(out, dim=1)

    def predict(self, x):
        out = self.predict_proba(x)
        if self.output_size == 1:
            return out > 0.5
        else:
            return out.argmax(dim=1)
