import torch
from ..utils import utils


def fit(data, network, **kwargs):
    loader = data["loader"]
    decoder = network["decoder"]
    encoder = network["encoder"]
    optimizer = network["optimizer"]
    criterion = network["criterion"]

    config = {"use_source_loader": False, "num_epochs": 100}
    config.update(kwargs)
    use_source_loader = config["use_source_loader"]
    num_epochs = config["num_epochs"]

    for _ in range(num_epochs):
        for X, y in loader:
            # Prep Data
            if use_source_loader:
                y = y[:, utils.COL_IDX_TASK]
            # Forward
            pred_y_task = decoder.predict_proba(encoder(X))
            if decoder.output_size > 1:
                y = y.to(torch.long)
            loss = criterion(pred_y_task, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update Params
            optimizer.step()

    return decoder, encoder
