import torch
from torch import nn, optim

from ..algo import algo
from .conv2d import Conv2d
from .mlp_decoder_three_layers import ThreeLayersDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dann:
    def __init__(
            self,
            domain_fc1_size: int,
            domain_fc2_size: int,
            task_fc1_size: int,
            task_fc2_size: int,
            output_size: int,
            input_size: int = 1600,
            lr_fc: float = 1e-4,
            lr_d: float = 1e-6,
            num_epochs: int = 100,
            device=torch.device("cpu"),
            is_target_weights: bool = False
        ):
        self.feature_extractor = Conv2d().to(device)
        self.task_classifier = ThreeLayersDecoder(input_size=input_size, output_size=output_size, fc1_size=task_fc1_size, fc2_size=task_fc2_size).to(device)
        self.domain_classifier = ThreeLayersDecoder(input_size=input_size, output_size=1, fc1_size=domain_fc1_size, fc2_size=domain_fc2_size).to(device)
        self.domain_criterion = nn.BCELoss()

        self.feature_otimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr_fc)
        self.domain_optimzier = optim.Adam(self.domain_classifier.parameters(), lr=lr_d)
        self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=lr_fc)
        self.num_ecochs = num_epochs
        self.device = device
        self.is_target_weights = is_target_weights
    
    def fit(self, source_loader, target_loader, test_target_X, test_target_y_task):
        self.feature_extractor, self.task_classifier, _ = algo.fit(
            source_loader,
            target_loader,
            test_target_X,
            test_target_y_task,
            self.feature_extractor,
            self.domain_classifier,
            self.task_classifier,
            self.domain_criterion,
            self.feature_otimizer,
            self.domain_optimzier,
            self.task_optimizer,
            num_epochs=self.num_ecochs,
            device=self.device,
            is_changing_lr=True,
            epoch_thr_for_changing_lr=11,
            changed_lrs=[1e-4, 1e-6],
            stop_during_epochs=True,
            epoch_thr_for_stopping=12,
            is_target_weights=self.is_target_weights
        )
    
    def predict(self, x):
        return self.task_classifier.predict(self.feature_extractor(x))
    
    def predict_proba(self, x):
        return self.task_classifier.predict_proba(self.feature_extractor(x))
    
    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()