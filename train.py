import torch
import torch.nn as nn
import numpy as np
import os

from model import *
from dataset import get_oasis_dataloaders


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "OASIS")
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-3
SAVE_PATH = "best_can_mode.pth"

def calculate_dice_score(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 4, smoothing: float = 1e-6) -> float:    
    """
    Computes macro-averaged Dice score across active classes.
    """
    predicted_classes = torch.argmax(predictions, dim=1)
    
    # One-hot encode predictions and targets: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    prediction_one_hot = torch.nn.functional.one_hot(predicted_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate overlap and cardinality per class across batch, height, and width
    intersection = (prediction_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    cardinality = prediction_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    
    # Compute dice per class with smoothing
    dice_per_class = (2.0 * intersection + smoothing) / (cardinality + smoothing)

    return dice_per_class.mean().item()

def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_dice += calculate_dice_score(outputs, targets, num_classes=num_classes) * batch_size
        total_samples += batch_size
        
        # Print current stats.
        current_avg_dice = running_dice / total_samples
        print(f"Batch [{batch_idx+1}/{len(dataloader)}] | Batch Loss: {loss.item():.4f} | Running Avg Dice: {current_avg_dice:.4f}")
        
    total_samples = len(dataloader.dataset)
    return running_loss / total_samples, running_dice / total_samples


def validate_one_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            running_dice += calculate_dice_score(outputs, targets) * images.size(0)

    total_samples = len(dataloader.dataset)
    return running_loss / total_samples, running_dice / total_samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Data loaders
    train_loader, val_loader, test_loader, num_classes = get_oasis_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    # Model. 4 Classes in the OASIS dataset.
    model = ContextAggregationModule(in_channels=1, num_classes=num_classes, C=64).to(device)
    
    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    best_val_dice = 0
    
    print("Starting Training Loop.")
    
    for epoch in range(EPOCHS):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, num_classes)
        
        scheduler.step(val_dice)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Save best model checkpoint based on validation Dice score
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"--> Saved new best model checkpoint to {SAVE_PATH} (Dice: {best_val_dice:.4f})")

    print("Training complete!")
    
if __name__ == "__main__":
    main()