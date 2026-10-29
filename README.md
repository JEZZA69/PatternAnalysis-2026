# 2D Context Aggregation Network vs. U-Net for OASIS Brain Tissue Segmentation
## Project Overview & Working Principles
### Problem Statement & Why Segmeentaion Matters
Automated neuroimaging analysis is critical for detecting progressive structural changes, such as ventricular enlargement and cortical atrophy in the brain. In this project, we tackle the OASIS 2D brain tissue segmentation challenge, which requires delineating structures like cortical gray matter, white matter, and cerebrospinal fluid (CSF) across 2D MRI slices. Precise anatomical contouring is essential, but it is complicated by intensity inhomogeneities (bias fields), partial volume effects at boundaries, and complex, fine gyral folds.

### How 2D CAN Works
Standard convolutional networks downsample feature maps, which can wash out fine spatial details. The 2D Context Aggregation Network (CAN) utilises dilated convolutions to systematically expand the network's receptive field without reducing spatial resolution or discarding high-resolution anatomical details. Furthermore, we implement identity weight initialisaiton tricks to ensure stable gradient flow through the multi-scale layers.

### Implementation
![Architecture Visualisation](visualisations/architecture_visualisation.drawio.png)
The CAN architecture features 8 layers structured as a dense prediction pipeline that preserves full 256x256 spatial resolution by matching padding to dilation factors for all 3x3 kernels. Layers 1 through 6 systematically aggregate multi-scale context by exponentially increasing dilation factors (1,1,2,4,8,16), followed by a 3x3 refinement layer (dilation 1), and a final 1x1 convolution layer that projects the intermediate channels (of size C=64) down to the 4 target segmentation classes. To overcome training instability and prevent initial boundary hallucinations, the module implements an identity initialization trick. By setting all weights to zero except for the exact center pixel ([1,1]) of matching channels—which is set to one—the network creates a clean signal pass-through during the first epoch. This prevents random noise scrambling, ensuring smooth gradient flow during backpropagation so the network can reliably train.


## Feasibility Report:
### Intended user
The intended user of the model are doctors, specifically hospital staff such as neuroradiologists and neuro-imaging researchers requiring automated, high-precision tissue segmentation templates for structural MRI scans.
### Purpose
To build, evaluate, and benchmark a 2D Context Aggragation Network (CAN) that automates anatomical contouring. In this context, segmentation refers to the process of splitting training data into masks to assist clinicians in structural analysis.
### Scope
The project operates on a focused scale using the 2D OASIS brain dataset (/home/groups/comp3710/OASIS), which contains preprocessed structural brain MRI slices and corresponding ground-truth segmentation masks.
### Acceptance criteria
 - **Dice score performance:** The primary quantitative metric for evaluating the performance of the model is the Sorensen-Dice similarity coefficient (DSC). The dice coefficient measures the spatial overlap between two binary masks. A score of 1.0 represents a perfect match, while 0 indicates no overlap. A target score of >0.90 (90%) across all tissue labels is mandated by the task set by the course.
 - **Baseline Performance:** The model must outperform a standard U-Net model under the same dataset conditions in order to evaluate if "incorporating advanced multi-scale context can genuinely eliminate these hazardous boundary and slice-transition errors compared to a 2D baseline."
 - **Operational constraints:** The model must run comfortably on workstation hardware or local GPUs. Given that patients sensitive medical data must be kept on the hospital's private network.
### Model Choice & Course Concepts
 - **Model Architecture**: The chosen architecture is a 2D Context Aggregation Network (CAN) on the OASIS dataset (easy tier). Instead of downsampling or pooling, it gains wider, multi-scale contextual information by systematically expanding the receptive field. This enables more context to be captured by the network, enabling greater precision on the pixel level, making it ideal for identifying fine details.
 - **Dataset Configuration**: The preprocessed OASIS 2D brain dataset (located on rangpur) consists of pre-segmented groups of train, validate, and test datasets. This separation enforces strict air gap separation to prevent critical data leakage and over-fitting.
 - The Oasis Brain 2D dataset consists of 2D MRI brain slices of several patients.

### Preliminary Feasibility evidence
 - **Empirical Validation**: Initial exploratory prototyping has already been successfully executed, with a working 2D CAN model trained and evaluated to achieve a Dice score exceeding 90% (>0.90).
 - Current Phase: Having verified that the architecture is viable and capable of meeting performance targets, the next immediate phase is implementing a standard 2D U-Net baseline. This will allow for a direct comparative analysis to answer the core course research dilemma: whether multi-scale context aggregation genuinely improves local boundary precision compared to a standard baseline.
 - After determining the performance of the 2D CAN model relative to the 2D U-Net baseline, the next step will be to develop another 2D CAN with a more complex dataset (HipMRI 2D slices.)
 - **Compute Resources:** Initial training and prototyping were executed successfully on a local high-end consumer gaming GPU (AMD Radeon 9070 xt) without requiring enterprise-grade hardware, proving the model's lightweight efficiency for the easy/normal tier.

### Risks, Budget & Fallback
 - **Clinical Safety**: Crucially, there must always be a qualified human in between the model and the patient. Given a Dice score of >90%, it is reasonable to expect the model to be accurate, but in medical imaging, deployment risks are severe. If a model hallucinates boundaries, creates disconnected tissue fragments, or misses a critical atrophy zone, it could lead to faulty diagnostic support or flawed volumetric tracking. Furthermore, over-fitting on training slices can cause catastrophic failure on unseen and unique, out-of-distribution patient scans.
 - **Computing Budget**: The initial pipeline and easy-tier models were trained locally on consumer hardware. However, scaling up to more complex configurations (such as deeper normal/hard variants or 3D volumes) will leverage the university's Rangpur HPC cluster using slurm.
 - **Workaround / Fallback**: If dilated convolution blocks introduce training instability, gridding artifacts, or fail to cleanly beat the baseline, the fallback plan is to pivot to an Improved 2D U-Net architecture utilizing standard residual skip connections to secure baseline compliance.

### In summary:
Utilising the preprocessed 2D OASIS dataset, the lightweight model aims to achieve a Sorensen-Dice similarity coefficient exceeding 0.90 while outperforming a standard U-Net baseline on local GPU hardware to ensure medical data privacy. Initial prototyping has successfully met this performance target, clearing the path to evaluate the model's multi-scale context aggregation against the baseline before expanding the architecture to a complex HipMRI dataset. Given the severe clinical safety risks of boundary hallucinations, a qualified human must always verify the outputs, while a fallback plan to an Improved 2D U-Net with residual skip connections is established if training instability or artifacts arise.

## Data Pipeline, Processing & Reproducibility

### Dataset Structure & Split Justification
The OASIS 2D brain dataset consists of paired single-channel PNG images and 4-class segmentation masks divided across pre-defined directories:
 - **Training Set**: 9,664 slices (~85%)
 - **Validation Set**: 1,120 slices (~10%)
 - **Test Set**: 544 slices (~5%)
The dataset enforces strict patient-level separation. All slices originating from a specific patient volume reside exclusively within a single partition. There is no overlap of patient ID numbers between any of the directories, ensuring no risk of cross-slice patient memorisation and ensuring no overfitting or leakage.

### Preprocessing Pipeline
1. **Image Normalisation:** Images are loaded as grayscale (`L`) and normalised to `[0, 1]` by dividing by 255.
2. **Mask Categorical Encoding:** Masks are integer-coded. The raw values are `{0, 85, 170, 255}`, and integer division by 85 maps them to class indices `{0, 1, 2, 3}`.
3. **Tensor Formatting:** Images are shaped `[1, H, W]` and masks `[H, W]` to match `CrossEntropyLoss`.
4. No data augmentation is currently applied. 
Note: While the model achieves a Dice score exceeding the required 0.90 on the provided split, the absence of augmentations (such as random rotations, scaling, or elastic deformations) elaves the network potentially vulnerable to overfitting or poor generalisation when exposed to MRI scans from different scanners with varying intensity profiles. Future work should introduce spatial and intensity augmentations to improve robustness.

### Training configuration
| Setting | Value |
|---|---|
| Batch size | 30 |
| Optimiser | AdamW (lr = 5e-5, weight decay = 1e-4) |
| Scheduler | ReduceLROnPlateau (mode = max on val Dice, factor 0.5, patience 3) |
| Loss | Multi-class Cross-Entropy with label smoothing $\epsilon$ = 0.05 |
| Gradient clipping | Global norm, max 1.0 |

### Dependencies & Reproducibility Setup
```# Environment (Python 3.12)
python --version        # 3.12.x
pip install torch==2.4.1 torch-directml==0.2.5.dev240914 pillow==12.3.0 numpy==2.5.4 matplotlib==3.11.2

# Repository Execution Instructions
# Ensure OASIS dataset is placed in root directory: ./OASIS/
python dataset.py   # Verifies data loader and split integrity
python train.py     # Set MODEL_NAME = "can" or "unet" in script settings
python predict.py   # Executes inference, generates quantitative metrics and visualization overlays      
```
Datasets and `.pth` weights are **not** committed to the repository, as required by the assessment. 

## Investigation of the Open Research Dilemma
### The Spatial Context vs Clinical Boundary Utility Dilemma
Standard global Sorensen-Dice similarity coefficiencts can hide major local boundary failures. In neuroimaging segmentatino, high global Dice scores often coexist with topologically invalid tissue disconnections along narrow cortical gyri or boundary erosion between Gray Matter and CSF. This investigation evaluates whether preserving full 256x256 spatial resolution via dilated context aggregation (CAN) genuinely eliminates boundary errors ocmpared to a pooling-based encoder-decoder (2D U-Net).

### Quantitative Benchmarking
Evaluated on a test split completely isolated from training and validation steps, the following data was recorded during inferencing of the test split (544 slices) (see [predict.py](predict.py)).
| Model     | Background    | CSF   | Grey matter   | White matter  | Mean Dice (tissue labels)     | Mean IoU (Tissue Labels)  |
|---                |---|---|---|---|---|---|
| Baseline U-Net    | 0.9760 | 0.6803 | 0.8289 | 0.9021 | 0.8038 | 0.6817 |
| CAN (C = 64)      | 0.9914 | 0.8646 | 0.9152 | 0.9527 | 0.9108 | 0.8383 |

Key Result: the 2D CAN surpasses the assessment threshold of >0.9 DICE score, with a DICE on tissue labels (excluding background) of 91.08%, yielding substantial gains in fine structures such as CSF, (+18.43%), and Gray Matter (+8.63%) relative to the baseline U-Net.

### Resource Profiling Table
| Efficiency Metric                 | Baseline 2D U-Net | 2D CAN (C=64) | Trade-Off Impact  |
|---                                |---                |---            |---                |
|Total Parameters|406852|222468|Can requires 56.60% fewer parameters, greatly reducing model storage size|
|Peak GPU VRAM (30 batch)           |2.1 GB             |1.0 GB         |CAN reduces VRAM consumption by 52.4%, as it avoids storing heavy multi-scale encoder skip-connection caches.            |
| Inference Latency (per slice)     |11.05 ms           |11.54 ms       |Negligible latency penalty (+4.4%), maintaining real-time throughput suitable for clinical workflows.                      |
| Full Resolution Feature Maps      |Downsampled to 128x128| Retained at 25x256|Preserves fine spatial boundaries and gyral folds, trading downsampling memory savings for explic pixel-level precision.                                                                                  |


## Results & Visualisations
### Learning Curves
![model learning curves](visualisations/learning_curves.png)

### Boundary-sensitive metrics
Global Dice can mask boundary errors, which is the core of the assessment's dilemma.


## Qualitative Error Autopsy & Spatial Difference Maps
### CAN: Ground truth vs predictions
![oasis CAN predictions](visualisations/oasis_can_predictions.png)
While the performance of the CAN is not as bad as the U-Net, there are still issues with edge detection


### UNET: Ground truch vs predictions
![oasis UNET predictions](visualisations/oasis_unet_predictions.png)
The limitations of the U-NET architecture can be seen at a glance, 


TODO: find out where this slots in
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

### Why the CAN uses full resolution
A U-Net trades resolution for context through pooling. The CAN instead spends its compute on full-resolution feature maps. This favours thin structures and boundary precision, at a cost in activation memory, as Section 3.4 showed.

### Identity initialisation
 
Layers 2–7 are initialised with all weights zero except the centre tap of matching channel pairs, which is set to 1; all biases are zero. Each such layer then computes $\mathrm{ReLU}(x)$ for non-negative inputs, so the untrained network is approximately an identity map from layer 2 onward. The first layer (random) and the output layer (random) are the only places where the input is transformed at initialisation.




## Engineering & Managerial Recommendation
 - Trade-off synthesis, the pros and cons of using the 
 - Summary of safety concernts

## Artificial Intelligence Usage Disclosure
Artificial intelligence was utilised in the production of this project. Artificial intelligence models, particularly the free version of gemini present in google search, was particularly useful for:
 - Ideation and consolodation
 - Boilerplate code generation
 - Debugging
 - Code cleaning and refactoring 
 - Matplotlib visualisation graph generation
Frequently, Gemini was used as a consult for the broader scope of the project, and was incremental in providing direction and enlightenment for the codebase, so while it was helpful in generating boilerplate, the insight and information gleamed from short chat sessions was essential during broad codebase formulation.

## Literature and Further Reading
 - F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” in Proc. ICLR,
2016.
 - O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image
segmentation,” in Proc. MICCAI, 2015, pp. 234–241.