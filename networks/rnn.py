from torch import nn


class ManyToOneRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (x, _) = self.rnn(x)
        x = x[-1, :, :]
        return x
