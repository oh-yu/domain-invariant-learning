import torch
from torch import nn
from tqdm import tqdm

from ..utils import utils
from dann_algo import ReverseGradient

def fit(data, network, **kwargs):
    # Args
    source_loader, target_loader, target_prime_loader = data["source_loader"], data["target_loader"], data["target_prime_loader"]
    target_prime_X, target_prime_y_task = data["target_prime_X"], data["target_prime_y_task"]
    feature_extractor, domain_classifier_dim1, domain_classifier_dim2, task_classifier = (
        network["feature_extractor"],
        network["domain_classifier_dim1"],
        network["domain_classifier_dim2"],
        network["task_classifier"],
    )
    criterion = network["criterion"]
    feature_optimizer, domain_optimizer_dim1, domain_optimizer_dim2, task_optimizer = (
        network["feature_optimizer"],
        network["domain_optimizer_dim1"],
        network["domain_optimizer_dim2"],
        network["task_optimizer"],
    )
    config = {
        "num_epochs": 1000,
        "is_target_weights": False,
    }
    config.update(kwargs)
    num_epochs, is_target_weights = config["num_epochs"], config["is_target_weights"]

    # Fit
    loss_tasks = []
    loss_task_evals = []
    loss_domain_dim1s = []
    loss_domain_dim2s = []
    reverse_grad = ReverseGradient.apply
    for epoch in tqdm(range(1, num_epochs.item()+1)):
        feature_extractor.train()
        task_classifier.train()
        for (source_X_batch, source_Y_batch), (target_X_batch, target_y_domain_batch), (target_prime_X_batch, target_prime_y_domain_batch) in zip(
            source_loader, target_loader, target_prime_loader
        ):
            # 0. Prep Data
            source_y_domain_batch = source_Y_batch[:, utils.COL_IDX_DOMAIN]
            source_y_task_batch = source_Y_batch[:, utils.COL_IDX_TASK]
            if task_classifier.output_size == 1:
                source_y_task_batch = source_y_task_batch.to(torch.float32)
            else:
                source_y_task_batch = source_y_task_batch.to(torch.long)

            # 1. Forward
            ## 1.1 Feature Extractor
            source_X_batch = feature_extractor(source_X_batch)
            target_X_batch = feature_extractor(target_X_batch)
            target_prime_X_batch = feature_extractor(target_prime_X_batch)

            ## 1.2.1 Domain Classifier Dim1
            source_X_batch_reversed_grad = reverse_grad(source_X_batch, epoch, num_epochs)
            target_X_batch = reverse_grad(target_X_batch, epoch, num_epochs)
            target_prime_X_batch = reverse_grad(target_prime_X_batch, epoch, num_epochs)

            pred_source_y_domain = domain_classifier_dim1(source_X_batch_reversed_grad)
            pred_target_y_domain_dim1 = domain_classifier_dim1(target_X_batch)
            pred_source_y_domain = torch.sigmoid(pred_source_y_domain).reshape(-1)
            pred_target_y_domain_dim1 = torch.sigmoid(pred_target_y_domain_dim1).reshape(-1)
            loss_domain_dim1 = criterion(pred_source_y_domain, source_y_domain_batch)
            loss_domain_dim1 += criterion(pred_target_y_domain_dim1, target_y_domain_batch)
            loss_domain_dim1s.append(loss_domain_dim1.item())

            ## 1.2.2 Domain Classifier Dim2
            pred_target_y_domain_dim2 = domain_classifier_dim2(target_X_batch)
            pred_target_prime_y_domain = domain_classifier_dim2(target_prime_X_batch)

            pred_target_y_domain_dim2  = torch.sigmoid(pred_target_y_domain_dim2).reshape(-1)
            pred_target_prime_y_domain  = torch.sigmoid(pred_target_prime_y_domain).reshape(-1)

            loss_domain_dim2 = criterion(pred_target_y_domain_dim2, target_y_domain_batch)
            loss_domain_dim2 += criterion(pred_target_prime_y_domain, target_prime_y_domain_batch)
            loss_domain_dim2s.append(loss_domain_dim2.item())

            loss_domain = loss_domain_dim1  + loss_domain_dim2
            ## 1.3 Task Classifier
            pred_y_task = task_classifier.predict_proba(source_X_batch)
            if task_classifier.output_size == 1:
                criterion_task = nn.BCELoss()
            else:
                criterion_task = nn.CrossEntropyLoss()
            loss_task = criterion_task(pred_y_task, source_y_task_batch)
            loss_tasks.append(loss_task.item())


            # 2. Backward
            feature_optimizer.zero_grad()
            domain_optimizer_dim1.zero_grad()
            domain_optimizer_dim2.zero_grad()
            task_optimizer.zero_grad()

            loss_domain.backward(retain_graph=True)
            loss_task.backward()
            # 3. Update Params

            feature_optimizer.step()
            domain_optimizer_dim1.step()
            domain_optimizer_dim2.step()
            feature_optimizer.step()

    # Eval
    feature_extractor.eval()
    task_classifier.eval()
    with torch.no_grad():
            target_prime_feature_eval = feature_extractor(target_prime_X)
            pred_y_task_eval = task_classifier.predict(target_prime_feature_eval)
            acc = sum(pred_y_task_eval == target_prime_y_task) / target_prime_y_task.shape[0]
    loss_task_evals.append(acc.item())
    
    print(f"Epoch: {epoch}, Loss Domain Dim1: {loss_domain_dim1}, Loss Domain Dim2: {loss_domain_dim2},  Loss Task: {loss_task}, Acc: {acc}")
    return feature_extractor, task_classifier
