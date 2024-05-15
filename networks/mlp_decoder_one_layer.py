import torch
import torch.nn.functional as F
from torch import nn


class OneLayerDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        return self.fc1(x)

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
