# Project Overview & Working Principles
## Problem Statement & Why Segmeentaion Matters
Automated neuroimaging analysis is critical for detecting progressive structural changes, such as ventricular enlargement and cortical atrophy in the brain. 
In this project, we tackle the OASIS 2D brain tissue segmentation challenge, which requires delineating structures like cortical gray matter, white matter, and cerebrospinal fluid (CSF) across 2D MRI slices. Precise anatomical contouring is essential, but it is complicated by intensity inhomogeneities (bias fields), partial volume effects at boundaries, and complex, fine gyral folds.
## How 2D CAN Works
Standard convolutional networks downsample feature maps, which can wash out fine spatial details. The 2D Context Aggregation Network (CAN) utilizes dilated convolutions to systematically expand the network's receptive field without reducing spatial resolution or discarding high-resolution anatomical details. Furthermore, we implement identity weight initializaiton tricks to ensure stable gradient flow through the multi-scale layers.
## Architecture Visualisation
![Architecture Visualisation](visualisations/architecture_visualisation.drawio.png)
The CAN architecture features 8 layers,

# Midpoint Feasibility Review
   ## Executive summary
 - See feasibility-review.md for the full feasibility review
# Data Pipeline & Reproducibility
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
 - Matplotlib visualization graph generation
Frequently, Gemini was used as a consult for the broader scope of the project, and was incremental in providing direction and enlightenment for the codebase.
# Literature and Further Reading
See https://arxiv.org/pdf/1511.07122 for more info on CAN architecture
