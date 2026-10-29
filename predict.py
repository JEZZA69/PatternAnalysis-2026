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
CLASS_COLORS = ['none', '#e6194b', '#3cb44b', '#4363d8']


def build_model(model_type, device):
    """Return (model, weights_filename, tag) for the chosen model type."""
    if model_type.lower() == "unet":
        model = Simple2DUNet(in_channels=1, num_classes=NUM_CLASSES, C=64)
        return model.to(device), "best_unet_model.pth", "unet"
    model = ContextAggregationModule(in_channels=1, num_classes=NUM_CLASSES, C=64)
    return model.to(device), "best_can_model.pth", "can"

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


def safe_ratio(num, den):
    """Return num/den, or NaN when the denominator is zero (class absent everywhere)."""
    return float(num) / float(den) if den > 0 else float("nan")


def evaluate(model, loader, device):
    """
    Runs inference on the full loader and returns a dict of metrics.
 
    Global counts per class c over the whole test set:
        I_c = |pred==c AND gt==c|     (intersection)
        P_c = |pred==c|               (prediction cardinality)
        T_c = |gt==c|                 (target cardinality)
 
        Dice_c = 2 I_c / (P_c + T_c)
        IoU_c  = I_c / (P_c + T_c - I_c)
    """
    inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    pred_card = np.zeros(NUM_CLASSES, dtype=np.int64)
    tgt_card = np.zeros(NUM_CLASSES, dtype=np.int64)
    per_slice_dice = []  # rows: slices; cols: classes (NaN where undefined)
 
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()   # [B, H, W]
            gts = targets.numpy()                                # [B, H, W]
 
            for p, g in zip(preds, gts):
                slice_dice = np.full(NUM_CLASSES, np.nan)
                for c in range(NUM_CLASSES):
                    pc = (p == c)
                    gc = (g == c)
                    i = np.logical_and(pc, gc).sum()
                    pn, tn = pc.sum(), gc.sum()
 
                    inter[c] += i
                    pred_card[c] += pn
                    tgt_card[c] += tn
 
                    slice_dice[c] = safe_ratio(2 * i, pn + tn)
                per_slice_dice.append(slice_dice)
 
    union = pred_card + tgt_card - inter
    dice = np.array([safe_ratio(2 * inter[c], pred_card[c] + tgt_card[c]) for c in range(NUM_CLASSES)])
    iou = np.array([safe_ratio(inter[c], union[c]) for c in range(NUM_CLASSES)])
 
    per_slice_dice = np.array(per_slice_dice)  # [N_slices, NUM_CLASSES]
 
    # Per-slice mean over tissue classes, skipping undefined (NaN) entries
    slice_tissue_mean = np.nanmean(per_slice_dice[:, TISSUE_CLASSES], axis=1)
 
    return {
        "dice": dice,
        "iou": iou,
        "mean_dice_all": np.nanmean(dice),
        "mean_iou_all": np.nanmean(iou),
        "mean_dice_tissue": np.nanmean(dice[TISSUE_CLASSES]),
        "mean_iou_tissue": np.nanmean(iou[TISSUE_CLASSES]),
        "slice_dice_mean": np.nanmean(slice_tissue_mean),
        "slice_dice_std": np.nanstd(slice_tissue_mean),
        "num_slices": len(per_slice_dice),
    }
    
def sanity_check_iou_dice(dice, iou, tol=1e-6):
    """
    Identity: IoU = Dice / (2 - Dice). Derivation:
        D = 2I/(U+I) with U = P+T-I  =>  D(U+I) = 2I  =>  DU = (2-D)I  =>  I/U = D/(2-D)
    If this fails, the Dice or IoU formula has a bug.
    """
    for c in range(NUM_CLASSES):
        if np.isnan(dice[c]):
            continue
        expected = dice[c] / (2 - dice[c])
        assert abs(iou[c] - expected) < tol, (
            f"Dice/IoU mismatch for class {c}: IoU={iou[c]:.6f}, D/(2-D)={expected:.6f}"
        )
    print("Sanity check passed: IoU = Dice / (2 - Dice) for every class.")

def fmt(x):
    return "n/a" if np.isnan(x) else f"{x:.4f}"
 
def print_and_save_report(metrics, tag):
    d, u = metrics["dice"], metrics["iou"]
 
    header = f"{'Class':<15}{'Dice':>10}{'IoU':>10}"
    print("\n=== Test set results ({} slices, model = {}) ===".format(metrics["num_slices"], tag.upper()))
    print(header)
    print("-" * len(header))
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:<15}{fmt(d[c]):>10}{fmt(u[c]):>10}")
    print("-" * len(header))
    print(f"{'Mean (all 4)':<15}{fmt(metrics['mean_dice_all']):>10}{fmt(metrics['mean_iou_all']):>10}")
    print(f"{'Mean (tissue)':<15}{fmt(metrics['mean_dice_tissue']):>10}{fmt(metrics['mean_iou_tissue']):>10}")
    print(f"\nPer-slice tissue Dice: {metrics['slice_dice_mean']:.4f} +/- {metrics['slice_dice_std']:.4f}")

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