import torch
import torch.nn as nn
import numpy as np


def calculate_dice_score(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 24, smoothing: float = 1e-6) -> float:    
    """
    Computes macro-averaged Dice score across active classes.
    """
    predicted_classes = torch.argmax(predictions, dim=1)
    
    # One-hot encode predictions and targets: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    pred_one_hot = torch.nn.functional.one_hot(predicted_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate overlap and cardinality per class across batch, height, and width
    intersection = (pred_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    cardinality = pred_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    
    # Compute dice per class with smoothing
    dice_per_class = (2.0 * intersection + smoothing) / (cardinality + smoothing)

    return dice_per_class.mean().item()