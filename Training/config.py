import torch

CONFIG = {
    "epochs": 100,  # Number of epochs for training
    "img_size": 384,  # Image size (height and width)
    "valid_batch_size": 8,  # Batch size for validation
    "learning_rate": 0.00008,  # Learning rate for optimizer
    "weight_decay": 0.0005,  # Weight decay for optimizer
    "n_accumulate": 1,  # Gradient accumulation steps
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Device to use for training (GPU if available)
}

