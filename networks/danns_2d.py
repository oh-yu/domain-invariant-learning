import torch
from torch import nn, optim

from .conv2d import Conv2d
from .conv1d_two_layers import Conv1dTwoLayers
from .conv1d_three_layers import Conv1dThreeLayers
from .mlp_decoder_three_layers import ThreeLayersDecoder
from ..algo.dann2D_algo import fit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Danns2D:
    def __init__(self, experiment: str):
        assert experiment in ["ECOdataset", "ECOdataset_synthetic", "HHAR", "MNIST"]
        if experiment in ["ECOdataset", "ECOdataset_synthetic"]:
            self.feature_extractor = Conv1dTwoLayers(input_size=3).to(DEVICE)
            self.domain_classifier_dim1 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.domain_classifier_dim2 = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)
            self.task_classifier = ThreeLayersDecoder(
                input_size=128, output_size=1, dropout_ratio=0, fc1_size=50, fc2_size=10
            ).to(DEVICE)

            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs = 200


        elif experiment == "HHAR":
            self.feature_extractor = Conv1dThreeLayers(input_size=6).to(DEVICE)
            self.domain_classifier_dim1 = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(
                DEVICE
            )
            self.domain_classifier_dim2 = ThreeLayersDecoder(input_size=128, output_size=1, dropout_ratio=0.3).to(
                DEVICE
            )
            self.task_classifier= ThreeLayersDecoder(input_size=128, output_size=6, dropout_ratio=0.3).to(DEVICE)
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs= 200
        
        elif experiment in ["MNIST"]:
            self.feature_extractor = Conv2d()
            self.task_classifier = ThreeLayersDecoder(
                input_size=1152, output_size=10, fc1_size=3072, fc2_size=2048
            )
            self.domain_classifier_dim1 = ThreeLayersDecoder(
                input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024
            )
            self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0001)
            self.domain_optimizer_dim1 = optim.Adam(self.domain_classifier_dim1.parameters(), lr=0.0001)
            self.task_optimizer = optim.Adam(self.task_classifier.parameters(), lr=0.0001)
            self.criterion = nn.BCELoss()
            self.num_epochs = 100

            self.domain_classifier_dim2 = ThreeLayersDecoder(
                input_size=1152, output_size=1, fc1_size=1024, fc2_size=1024
            )
            self.domain_optimizer_dim2 = optim.Adam(self.domain_classifier_dim2.parameters(), lr=0.0001)

    
    def fit(self, source_loader, target_loader, target_prime_loader, test_target_prime_X, test_target_prime_y_task):
        data = {
            "source_loader": source_loader,
            "target_loader": target_loader,
            "target_prime_loader": target_prime_loader,
            "target_prime_X": test_target_prime_X,
            "target_prime_y_task": test_target_prime_y_task,
        }
        network = {
            "feature_extractor": self.feature_extractor,
            "domain_classifier_dim1": self.domain_classifier_dim1,
            "domain_classifier_dim2": self.domain_classifier_dim2,
            "task_classifier": self.task_classifier,
            "criterion": self.criterion,
            "feature_optimizer": self.feature_optimizer,
            "domain_optimizer_dim1": self.domain_optimizer_dim1,
            "domain_optimizer_dim2": self.domain_optimizer_dim2,
            "task_optimizer": self.task_optimizer,
        }
        config = {
            "num_epochs": self.num_epochs
        }
        self.feature_extractor, self.task_classifier, acc = fit(data, network, **config)
        return acc