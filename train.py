import torch
import torch.nn as nn
import numpy as np


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