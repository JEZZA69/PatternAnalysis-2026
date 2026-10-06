import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import nibabel as nib

# OASIS Brain 2D dataset. 
# Found here: https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md

class OASIS2DDataset(Dataset):
    def __init__(self, subject_dirs):
        self.subject_dirs = sorted(subject_dirs)

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        sub_dir = self.subject_dirs[idx]
        
        # Define exact paths to the 2d scan and its 24-structure segmentation labels
        img_path = os.path.join(sub_dir, "slice_norm.nii.gz")
        mask_path = os.path.join(sub_dir, "slice_seg24.nii.gz")
        
        # 2. Load NIfTI volumes and extract raw numpy arrays
        # .get_fdata() converts the data into float arrays automatically
        image_np = nib.load(img_path).get_fdata()
        mask_np = nib.load(mask_path).get_fdata()
        
        # Convert to tensor
        # FloatTensor with a channel dimension added -> [1, Height, Width]
        image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)
        
        # Mask: LongTensor (integers 0-23 for classification loss tasks) -> [Height, Width]
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)
        
        return image_tensor, mask_tensor

def get_oasis_dataloaders(data_dir, batch_size=16, train_split=0.7, val_split=0.15):
    # Find all subject folders matching pattern 'OASIS_OAS1_*'
    all_subjects = glob.glob(os.path.join(data_dir, "OASIS_OAS1_*"))
    all_subjects.sort()  # Keep reproducible order
    
    # Calculate patient-level structural splits 
    num_subs = len(all_subjects)
    train_end = int(num_subs * train_split)
    val_end = int(num_subs * (train_split + val_split))
    
    # Split directories cleanly (no leakage, one patient belongs strictly to one set)
    train_dirs = all_subjects[:train_end]
    val_dirs = all_subjects[train_end:val_end]
    test_dirs = all_subjects[val_end:]
    
    # Create dataset objects using our custom medical reader
    train_dataset = OASIS2DDataset(train_dirs)
    val_dataset = OASIS2DDataset(val_dirs)
    test_dataset = OASIS2DDataset(test_dirs)
    
    # Bundle into final PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader