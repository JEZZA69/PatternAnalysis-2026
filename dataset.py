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
    def __init__(self, images_dir, masks_dir):
        """
        Directly loads paired images and masks from their respective pre-split folders.
        """
        self.image_paths = sorted(glob.glob(os.path.join(images_dir)))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir)))
        
        # Safety check to ensure images match masks 1:1
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch! Found {len(self.image_paths)} images in {os.path.basename(images_dir)} " \
            f"but {len(self.mask_paths)} masks in {os.path.basename(masks_dir)}."

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 1. Open the 2D image and mask as grayscale
        img = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        # 2. Convert to raw numpy formats
        img_np = np.array(img, dtype=np.float32) / 255.0  # Scale pixel intensities to [0, 1]
        mask_np = np.array(mask, dtype=np.int64)          # Class labels as integers
        
        # 3. Handle UQ thresholding: Convert grayscale label intensity to binary classes (0 and 1)
        if mask_np.max() > 1:
            mask_np = (mask_np > 127).astype(np.int64)

        # 4. Convert into standard PyTorch Tensors
        image_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0) # [1, H, W]
        mask_tensor = torch.tensor(mask_np, dtype=torch.long)                 # [H, W]
        
        return image_tensor, mask_tensor

def get_oasis_dataloaders(data_dir, batch_size=16):
    """
    Hooks directly into the pre-split UQ folder directories and yields 
    production-ready PyTorch loaders.
    """
    # Map the exact directory paths from your local machine
    train_img_dir = os.path.join(data_dir, "keras_png_slices_train")
    train_msk_dir = os.path.join(data_dir, "keras_png_slices_seg_train")
    
    val_img_dir = os.path.join(data_dir, "keras_png_slices_validate")
    val_msk_dir = os.path.join(data_dir, "keras_png_slices_seg_validate")
    
    test_img_dir = os.path.join(data_dir, "keras_png_slices_test")
    test_msk_dir = os.path.join(data_dir, "keras_png_slices_seg_test")
    
    # Initialize separate datasets utilizing pre-made splits
    train_ds = OASIS2DDataset(train_img_dir, train_msk_dir)
    val_ds = OASIS2DDataset(val_img_dir, val_msk_dir)
    test_ds = OASIS2DDataset(test_img_dir, test_msk_dir)
    
    print(f"--> [Dataset Ingestion] Train samples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    # Wrap with robust PyTorch loaders
    train_loader = DataLoader(train_dataset=train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset=val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset=test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader



# ==========================================
# SANITY CHECK TEST BLOCK (AI generated.)
# ==========================================
if __name__ == "__main__":
    # Tests layout assuming terminal execution from inside the project directory root
    LOCAL_DATA_DIR = "./OASIS"
    try:
        tl, vl, tstl = get_oasis_dataloaders(LOCAL_DATA_DIR, batch_size=4)
        imgs, msks = next(iter(tl))
        print("\n=== Validation Check Successful ===")
        print(f"Loaded Image Tensor Batch Shape: {imgs.shape} -> Expected [4, 1, H, W]")
        print(f"Loaded Mask Tensor Batch Shape:  {msks.shape} -> Expected [4, H, W]")
    except Exception as e:
        print(f"\n[Verification Error]: Validation failed. Check directory path name. Error: {e}")