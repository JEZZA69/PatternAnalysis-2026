# 2D Context Aggregation Network vs. U-Net for OASIS Brain Tissue Segmentation
## Project Overview & Working Principles
### Problem Statement & Why Segmeentaion Matters
Automated neuroimaging analysis is critical for detecting progressive structural changes, such as ventricular enlargement and cortical atrophy in the brain. In this project, we tackle the OASIS 2D brain tissue segmentation challenge, which requires delineating structures like cortical gray matter, white matter, and cerebrospinal fluid (CSF) across 2D MRI slices. Precise anatomical contouring is essential, but it is complicated by intensity inhomogeneities (bias fields), partial volume effects at boundaries, and complex, fine gyral folds.

### How 2D CAN Works
Standard convolutional networks downsample feature maps, which can wash out fine spatial details. The 2D Context Aggregation Network (CAN) utilises dilated convolutions to systematically expand the network's receptive field without reducing spatial resolution or discarding high-resolution anatomical details. Furthermore, we implement identity weight initialisaiton tricks to ensure stable gradient flow through the multi-scale layers.

### Implementation
![Architecture Visualisation](visualisations/architecture_visualisation.drawio.png)
The CAN architecture features 8 layers structured as a dense prediction pipeline that preserves full 256x256 spatial resolution by matching padding to dilation factors for all 3x3 kernels. Layers 1 through 6 systematically aggregate multi-scale context by exponentially increasing dilation factors (1,1,2,4,8,16), followed by a 3x3 refinement layer (dilation 1), and a final 1x1 convolution layer that projects the intermediate channels (of size C=64) down to the 4 target segmentation classes. To overcome training instability and prevent initial boundary hallucinations, the module implements an identity initialization trick. By setting all weights to zero except for the exact center pixel ([1,1]) of matching channels—which is set to one—the network creates a clean signal pass-through during the first epoch. This prevents random noise scrambling, ensuring smooth gradient flow during backpropagation so the network can reliably train.

## Feasibility Report
The full feasability review can be found here: ![feasibility-review.md](feasibility-review.md)
### In summary:
Utilising the preprocessed 2D OASIS dataset, the lightweight model aims to achieve a Sorensen-Dice similarity coefficient exceeding 0.90 while outperforming a standard U-Net baseline on local GPU hardware to ensure medical data privacy. Initial prototyping has successfully met this performance target, clearing the path to evaluate the model's multi-scale context aggregation against the baseline before expanding the architecture to a complex HipMRI dataset. Given the severe clinical safety risks of boundary hallucinations, a qualified human must always verify the outputs, while a fallback plan to an Improved 2D U-Net with residual skip connections is established if training instability or artifacts arise.

## Data Pipeline & Reproducibility
 ### Dataset and Splits
 - Source: preprocessed OASIS 2D slices `1keras_png_slices_*` folders, with paired masks in `keras_png_slices_seg_`. This dataset was downloaded from the UQ Rangpur cluster and underwent zero preprocessing before use in this repository.
 - Splits: the provided train / validate / test folders are used as-is. The splits for training, validation, and testing are roughly 85%, 10%, and 5%  (9664,1120,544 png files respectively)
 - Patient-level seperation has been handled by the provided dataset. There is no overlap of patient ID numbers between the train/validation/test datasets. 

### Preprocessing
1. Images are loaded as grayscale (`L`) and normalised to `[0, 1]` by dividing by 255.
2. Masks are integer-coded. The raw values are `{0, 85, 170, 255}`, and integer division by 85 maps them to class indices `{0, 1, 2, 3}`.
3. Images are shaped `[1, H, W]` and masks `[H, W]` to match `CrossEntropyLoss`.
4. No data augmentation is currently applied. 
Note: While the model achieves a Dice score exceeding the required 0.90 on the provided split, the absence of augmentations (such as random rotations, scaling, or elastic deformations) elaves the network potentially vulnerable to overfitting or poor generalisation when exposed to MRI scans from different scanners with varying intensity profiles. Future work should introduce spatial and intensity augmentations to improve robustness.

### Training configuration
| Setting | Value |
|---|---|
| Batch size | 30 |
| Epochs | 10 per run, resumable from checkpoint |
| Optimiser | AdamW, lr = 5e-5, weight decay = 1e-4 |
| Scheduler | ReduceLROnPlateau (mode = max on val Dice, factor 0.5, patience 3) |
| Loss | Cross-entropy with label smoothing ε = 0.05 |
| Gradient clipping | Global norm, max 1.0 |

### Reproducing the results
```# Environment (exact versions required by the assessment)
python --version        # TODO[e.g. 3.11.x]
pip install torch==2.4.1 torch-directml==0.2.5.dev240914 pillow==12.3.0 numpy==2.5.4 matplotlib==3.11.2

# Data: place the OASIS folder in the project root so that ./OASIS/ contains the six keras_png_slices_* folders
python dataset.py       
python train.py         # set MODEL_NAME = "can" or "unet" in train.py first
python predict.py       
```
Datasets and `.pth` weights are **not** committed to the repository, as required by the assessment. 

## Models
### 2D Context Aggregation Network (CAN)
| Layer | Operation | Kernel | Dilation | Padding | Output channels |
|---|---|---|---|---|---|
| 1 | Conv + ReLU | 3×3 | 1 | 1 | 64 |
| 2 | Conv + ReLU | 3×3 | 1 | 1 | 64 |
| 3 | Conv + ReLU | 3×3 | 2 | 2 | 64 |
| 4 | Conv + ReLU | 3×3 | 4 | 4 | 64 |
| 5 | Conv + ReLU | 3×3 | 8 | 8 | 64 |
| 6 | Conv + ReLU | 3×3 | 16 | 16 | 64 |
| 7 | Conv + ReLU | 3×3 | 1 | 1 | 64 |
| 8 | Conv (logits) | 1×1 | — | 0 | 4 |  

For each 3×3 layer, padding equals dilation, so the spatial size stays at 256×256 (the output size of a convolution is $H + 2p - d(k-1)$, which equals $H$ when $p = d$ and $k = 3$). 

### Baseline 2D U-Net (`Simple2DUNet`, C = 64)
The baseline has one encoder level, one max-pool, one bilinear upsample and a skip connection, with 406,852 parameters. This is a *shallow* U-Net, not the Improved U-Net of Isensee et al. The assessment's Easy-difficulty baseline is stated in terms of a standard model, so this is a valid comparison, but the conclusions apply to this shallow baseline specifically and should be stated that way in the results.

## Quantitative Benchmarking

## Qualitative Error Autopsy & Spatial Difference Maps
## Engineering & Managerial Recommendation
 - Trade-off synthesis, the pros and cons of using the 
 - Summary of safety concernts

## Artificial Intelligence Usage Disclosure
Artificial intelligence was utilised extensively in the production of this project. Artificial intelligence models, particularly the free version of gemini present in google search, was extensively used for:
 - Ideation and consolodation
 - Boilerplate code generation
 - Debugging
 - Code cleaning and refactoring 
 - Matplotlib visualisation graph generation
Frequently, Gemini was used as a consult for the broader scope of the project, and was incremental in providing direction and enlightenment for the codebase.

## Literature and Further Reading
 - F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” in Proc. ICLR,
2016.
 - O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image
segmentation,” in Proc. MICCAI, 2015, pp. 234–241.