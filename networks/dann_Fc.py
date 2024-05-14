import torch
from torch import nn, optim

from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder
from ..utils import utils


class Dann_F_C(nn.Module):
    def __init__(self, input_size: int=1152, output_size: int = 10, fc1_size: int = 3072, fc2_size: int = 2048, device = torch.device("cpu")):
        super().__init__()
        self.conv2d = Conv2d().to(device)
        self.decoder = ThreeLayersDecoder(input_size=input_size, output_size=output_size, fc1_size=fc1_size, fc2_size=fc2_size).to(device)
        self.device = device
        self.optimizer = optim.Adam(list(self.conv2d.parameters())+list(self.decoder.parameters()), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs=10
    
    def fit_without_adapt(self, source_loader):
        for _ in range(self.num_epochs):
            for source_X_batch, source_Y_batch in source_loader:
                # Prep Data
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]

                # Forward
                pred_y_task = self.decoder(self.conv2d(x))
                if self.decoder.output_size == 1:
                    pred_y_task = torch.sigmoid(pred_y_task).reshape(-1)
                else:
                    source_y_task_batch = source_y_task_batch.to(torch.long)
                    pred_y_task = torch.softmax(pred_y_task, dim=1)
                
                loss_task = self.criterion(pred_y_task, source_y_task_batch)

                # Backward
                self.optimizer.zero_grad()
                loss_task.backward()

                # Updata Params
                self.optimizer.step()
    def fit_on_target(self, train_target_prime_loader):
        # Fit
        for _ in range(10):
            for X, y in train_target_prime_loader:
                self.optimizer.zero_grad()
                pred_y_task = self.predict_proba(X)
                loss = self.criterion(pred_y_task, y[:, 0].to(torch.long))
                loss.backward()
                self.optimizer.step()
    
    def forward(self, x):
        return self.decoder(self.conv2d(x))
    
    def predict(self, x):
        return self.decoder.predict(self.conv2d(x))
    
    def predict_proba(self, x):
        return self.decoder.predict_proba(self.conv2d(x))