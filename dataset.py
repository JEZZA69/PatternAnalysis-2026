import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class OASISDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_oasis_dataloaders(data_dir, batch_size=16, train_split=0.7, val_split=0.15):
    """
    Scans data_dir, performs patient-level splitting, and returns DataLoaders.
    """
    # 1. Gather file paths
    all_images = glob.glob(os.path.join(data_dir, "images", "*.png"))
    
    # 2. Extract patient IDs to enforce patient-level split (prevent leakage)
    patient_ids = list(set([os.path.basename(f).split('_')[0] for f in all_images]))
    patient_ids.sort()

    num_patients = len(patient_ids)
    train_end = int(num_patients * train_split)
    val_end = int(num_patients * (train_split + val_split))

    train_patients = set(patient_ids[:train_end])
    val_patients = set(patient_ids[train_end:val_end])
    test_patients = set(patient_ids[val_end:])

    train_imgs = [f for f in all_images if os.path.basename(f).split('_')[0] in train_patients]
    val_imgs = [f for f in all_images if os.path.basename(f).split('_')[0] in val_patients]
    test_imgs = [f for f in all_images if os.path.basename(f).split('_')[0] in test_patients]

    # Map image paths to corresponding mask paths
    train_masks = [f.replace("images", "masks") for f in train_imgs]
    val_masks = [f.replace("images", "masks") for f in val_imgs]
    test_masks = [f.replace("images", "masks") for f in test_imgs]

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = OASISDataset(train_imgs, train_masks, transform=transform)
    val_dataset = OASISDataset(val_imgs, val_masks, transform=transform)
    test_dataset = OASISDataset(test_imgs, test_masks, transform=transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader