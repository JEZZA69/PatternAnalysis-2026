import torch
import torch.nn as nn
import numpy as np


<<<<<<< HEAD
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
=======
def calculate_dice_score(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 24) -> float:
    predicted_classes = torch.argmax(predictions, dim=1)
    total_dice = 0.0
    
    for class_id in range(num_classes):
        predicted_mask = (predicted_classes == class_id)
        target_mask = (targets == class_id)
        
        intersection = (predicted_mask & target_mask).sum().float()
        cardinality = predicted_mask.sum().float() + target_mask.sum().float()
        
        # If a class is completely absent from both masks, it represents a perfect true negative.
        if cardinality == 0:
            class_dice = 1.0
        else:
            class_dice = (2.0 * intersection) / cardinality
            
        total_dice += class_dice
        
    return total_dice / num_classes
>>>>>>> f2f2b4dbb5ba1d6db1555ebf5d3875e810f66b51
