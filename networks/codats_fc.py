import torch
from torch import nn, optim

from ..utils import utils
from .conv1d_three_layers import Conv1dThreeLayers
from .conv1d_two_layers import Conv1dTwoLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoDATS_F_C(nn.Module):
    def __init__(self, experiment: str):
        super().__init__()
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR"]
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.conv1d = Conv1dTwoLayers(input_size=3).to(DEVICE)
            self.decoder = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.optimizer = optim.Adam(list(self.conv1d.parameters()) + list(self.decoder.parameters()), lr=1e-4)
            self.criterion = nn.BCELoss()
            self.num_epochs = 300

        elif experiment == "HHAR":
            self.conv1d = Conv1dThreeLayers(input_size=6).to(DEVICE)
            self.decoder = ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)
            self.optimizer = optim.Adam(list(self.conv1d.parameters()) + list(self.decoder.parameters()), lr=1e-4)
            self.criterion = nn.CrossEntropyLoss()
            self.num_epochs = 300

    def fit_without_adapt(self, source_loader):
        for _ in range(self.num_epochs):
            for source_X_batch, source_Y_batch in source_loader:
                # Prep Data
                source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]

                # Forward
                pred_y_task = self.decoder(self.conv1d(source_X_batch))
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

    def fit_on_target(self, target_prime_loader):
        for _ in range(self.num_epochs):
            for target_prime_X_batch, target_prime_y_task_batch in target_prime_loader:
                pred_y_task = self.predict_proba(target_prime_X_batch)
                loss = self.criterion(pred_y_task, target_prime_y_task_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def forward(self, x):
        return self.decoder(self.conv1d(x))

    def predict(self, x):
        return self.decoder.predict(self.conv1d(x))

    def predict_proba(self, x):
        return self.decoder.predict_proba(self.conv1d(x))
