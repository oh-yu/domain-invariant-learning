

def fit_coral(source_loader, target_loader, num_epochs, task_classifier, criterion, optimizer, is_psuedo_weights):
    for epoch in range(1, num_epochs.item() + 1):
        task_classifier.train()
        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(
            source_loader, target_loader
        ):
            # 0. Data
            # 1. Forward
            # 1.1 Task Loss
            # 1.2 CoRAL Loss

            # 2. Backward
            # 3. Update Params
            pass
        # 4. Eval