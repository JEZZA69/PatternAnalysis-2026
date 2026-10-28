# 2D Context Aggregation Network vs. U-Net for OASIS Brain Tissue Segmentation
# Project Overview & Working Principles
## Problem Statement & Why Segmeentaion Matters
Automated neuroimaging analysis is critical for detecting progressive structural changes, such as ventricular enlargement and cortical atrophy in the brain. 
In this project, we tackle the OASIS 2D brain tissue segmentation challenge, which requires delineating structures like cortical gray matter, white matter, and cerebrospinal fluid (CSF) across 2D MRI slices. Precise anatomical contouring is essential, but it is complicated by intensity inhomogeneities (bias fields), partial volume effects at boundaries, and complex, fine gyral folds.
## How 2D CAN Works
Standard convolutional networks downsample feature maps, which can wash out fine spatial details. The 2D Context Aggregation Network (CAN) utilises dilated convolutions to systematically expand the network's receptive field without reducing spatial resolution or discarding high-resolution anatomical details. Furthermore, we implement identity weight initialisaiton tricks to ensure stable gradient flow through the multi-scale layers.
## Implementation
![Architecture Visualisation](visualisations/architecture_visualisation.drawio.png)
The CAN architecture features 8 layers structured as a dense prediction pipeline that preserves full 256x256 spatial resolution by matching padding to dilation factors for all 3x3 kernels. Layers 1 through 6 systematically aggregate multi-scale context by exponentially increasing dilation factors (1,1,2,4,8,16), followed by a 3x3 refinement layer (dilation 1), and a final 1x1 convolution layer that projects the intermediate channels (of size C=64) down to the 4 target segmentation classes. To overcome training instability and prevent initial boundary hallucinations, the module implements an identity initialization trick. By setting all weights to zero except for the exact center pixel ([1,1]) of matching channels—which is set to one—the network creates a clean signal pass-through during the first epoch. This prevents random noise scrambling, ensuring smooth gradient flow during backpropagation so the network can reliably train.

# Feasibility Report
The full feasability review can be found: ![here](feasibility-review.md)
## In summary:
 Utilising the preprocessed 2D OASIS dataset, the lightweight model aims to achieve a Sorensen-Dice similarity coefficient exceeding 0.90 while outperforming a standard U-Net baseline on local GPU hardware to ensure medical data privacy. Initial prototyping has successfully met this performance target, clearing the path to evaluate the model's multi-scale context aggregation against the baseline before expanding the architecture to a complex HipMRI dataset. Given the severe clinical safety risks of boundary hallucinations, a qualified human must always verify the outputs, while a fallback plan to an Improved 2D U-Net with residual skip connections is established if training instability or artifacts arise.
# Data Pipeline & Reproducibility
The 2D dataset
# Quantitative Benchmarking
# Qualitative Error Autopsy & Spatial Difference Maps
# Engineering & Managerial Recommendation
 - Trade-off synthesis, the pros and cons of using the 
 - Summary of safety concernts
# Artificial Intelligence Usage Disclosure
Artificial intelligence was utilised extensively in the production of this project. Artificial intelligence models, particularly the free version of gemini present in google search, was extensively used for:
 - Ideation and consolodation
 - Boilerplate code generation
 - Debugging
 - Code cleaning and refactoring 
 - Matplotlib visualisation graph generation
Frequently, Gemini was used as a consult for the broader scope of the project, and was incremental in providing direction and enlightenment for the codebase.
# Literature and Further Reading
 - F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” in Proc. ICLR,
2016.
 - O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image
segmentation,” in Proc. MICCAI, 2015, pp. 234–241.