def fit_dan(source_loader, target_loader, num_epochs,
            feature_extractor, task_classifier, domain_classifier):
    for epoch in range(1, num_epochs.item() + 1):
        feature_extractor.train()
        task_classifier.train()

        for (source_X_batch, source_Y_batch), (target_X_batch, _) in zip(
            source_loader, target_loader
        ):
            pass
            # 0. Data

            # 1. Forward

            # 1.1 Feature Extractor
            # 1.2 Task Classifier

            # 1.3 Task Loss
            # 1.4 MMD Loss

            # 2. Backward
            # 3. Update Params
        
        # 4. Eval
