import torch
from torch import nn, optim

from ..algo import algo
from .conv2d import Conv2d
from .mlp_decoder_domain import DomainDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dann:
    def __init__(self, domain_fc1_size: int, domain_fc2_size: int, task_fc1_size: int, task_fc2_size: int, output_size: int, input_size: int = 1600, lr: int = 1e-3, num_epochs: int = 100):
        self.feature_extractor = Conv2d().to(DEVICE)
        self.task_classifier = DomainDecoder(input_size=input_size, output_size=output_size, fc1_size=task_fc1_size, fc2_size=task_fc2_size).to(DEVICE)
        self.domain_classifier = DomainDecoder(input_size=input_size, output_size=1, fc1_size=domain_fc1_size, fc2_size=domain_fc2_size).to(DEVICE)
        self.domain_criterion = nn.BCELoss()

        self.feature_otimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        self.domain_optimzier = optim.Adam(self.domain_classifier.parameters(), lr=lr)
        self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=lr)
        self.num_ecochs = num_epochs
    
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
            num_epochs=self.num_ecochs
        )
    
    def predict(self, x):
        return self.task_classifier.predict(self.feature_extractor(x))
    
    def set_eval(self):
        self.task_classifier.eval()
        self.feature_extractor.eval()