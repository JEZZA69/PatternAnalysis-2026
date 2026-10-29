import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Import modules and dataset loader
from modules import ContextAggregationModule, Simple2DUNet
from dataset import get_oasis_dataloaders

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "OASIS")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "visualisations")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_TYPE = "can"      # 'can' or 'unet'
BATCH_SIZE = 8
NUM_CLASSES = 4
CLASS_NAMES = ["Background", "CSF", "Grey Matter", "White Matter"]
TISSUE_CLASSES = [1, 2, 3]  # exclude background for the tissue-only means


# Custom color palette for OASIS 4-class segmentation
# Class 0: Background (Transparent), Class 1: CSF, Class 2: Gray Matter, Class 3: White Matter
CLASS_COLORS = ['none', '#e6194b', '#3cb44b', '#4363d8']
CLASS_NAMES = ['Background', 'CSF', 'Gray Matter', 'White Matter']


def build_model(model_type, device):
    """Return (model, weights_filename, tag) for the chosen model type."""
    if model_type.lower() == "unet":
        model = Simple2DUNet(in_channels=1, num_classes=NUM_CLASSES, C=64)
        return model.to(device), "best_unet_model.pth", "unet"
    model = ContextAggregationModule(in_channels=1, num_classes=NUM_CLASSES, C=64)
    return model.to(device), "best_can_model.pth", "can"
 

def create_overlay_cmap():
    """Returns a ListedColormap for discrete class segmentation masks."""
    return ListedColormap(CLASS_COLORS)

def visualise_samples(model=None, dataloader=None, device='cpu', num_samples=4, save_filename='oasis_predictions.png'):
    """
    Extracts samples from the dataloader and generates side-by-side plots of:
    1. Raw Grayscale Input MRI
    2. Ground Truth Segmentation Mask
    3. Model Predicted Segmentation Mask
    """
    images, targets = next(iter(dataloader))
    num_samples = min(num_samples, images.size(0))

    if model is not None:
        model.eval()
        with torch.no_grad():
            outputs = model(images[:num_samples].to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        ncols = 3
    else:
        preds = None
        ncols = 2

    fig, axes = plt.subplots(num_samples, ncols, figsize=(4 * ncols, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = create_overlay_cmap()

    for idx in range(num_samples):
        img = images[idx, 0].numpy()
        gt = targets[idx].numpy()

        # 1. Input MRI Slice
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f"Sample {idx + 1}: Raw MRI")
        axes[idx, 0].axis('off')

        # 2. Ground Truth Overlay (Mask out background class 0 for transparency)
        axes[idx, 1].imshow(img, cmap='gray')
        gt_masked = np.ma.masked_where(gt == 0, gt)
        axes[idx, 1].imshow(gt_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
        axes[idx, 1].set_title(f"Sample {idx + 1}: Ground Truth")
        axes[idx, 1].axis('off')

        # 3. Model Prediction Overlay (Mask out background class 0 for transparency)
        if model is not None and preds is not None:
            pred = preds[idx]
            axes[idx, 2].imshow(img, cmap='gray')
            pred_masked = np.ma.masked_where(pred == 0, pred)
            axes[idx, 2].imshow(pred_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
            axes[idx, 2].set_title(f"Sample {idx + 1}: Prediction")
            axes[idx, 2].axis('off')

    # Add shared color legend at the bottom
    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(1, len(CLASS_COLORS))]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches), bbox_to_anchor=(0.5, -0.02), frameon=True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Saved visual output to: {save_path}")

def load_model_weights(model, weights_filename, device='cpu'):
    """Safely loads model weights, checking both script directory and project root."""
    
    weights_path = os.path.join(MODELS_DIR, weights_filename) # Inside models/ folder (matches train.py)    

    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        if 'model_state' in state_dict:
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {weights_path}")
    else:
        print(f"WARNING: Weights file '{weights_filename}' not found. Running with uninitialized weights.")
    return model

if __name__ == "__main__":
    from train import get_device
    device = get_device()
    print(f"Using inference device: {device}")

    # Load test/validation dataloader
    _, _, test_loader, num_classes = get_oasis_dataloaders(DATA_DIR, batch_size=8)
    
    if MODEL_TYPE.lower() == "unet":
        model = Simple2DUNet(in_channels=1, num_classes=num_classes, C=64).to(device)
        weights_file = "best_unet_model.pth"
        output_image_name = "oasis_unet_predictions.png"
    else:
        model = ContextAggregationModule(in_channels=1, num_classes=num_classes, C=64).to(device)
        weights_file = "best_can_model.pth"
        output_image_name = "oasis_can_predictions.png"

    # Load weights robustly
    model = load_model_weights(model, weights_file, device=device)

    # Run inference and save multi-panel visualisations
    visualise_samples(
        model=model,
        dataloader=test_loader,
        device=device,
        num_samples=4,
        save_filename=output_image_name
    )