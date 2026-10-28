
# Feasibility Review:
## User Need, Scope & Criteria
### Intended user
The intended user of the model are doctors, specifically hospital staff such as neuroradiologists and neuro-imaging researchers requiring automated, high-precision tissue segmentation templates for structural MRI scans.
### Purpose
To build, evaluate, and benchmark a 2D Context Aggragation Network (CAN) that automates anatomical contouring. In this context, segmentation refers to the process of splitting training data into masks to assist clinicians in structural analysis.
### Scope
The project operates on a focused scale using the 2D OASIS brain dataset (/home/groups/comp3710/OASIS), which contains preprocessed structural brain MRI slices and corresponding ground-truth segmentation masks.
### Acceptance criteria
 - Dice score performance: The primary quantitative metric for evaluating the performance of the model is the Sorensen-Dice similarity coefficient (DSC). The dice coefficient measures the spatial overlap between two binary masks. A score of 1.0 represents a perfect match, while 0 indicates no overlap. A target score of >0.90 (90%) across all tissue labels is mandated by the task set by the course.
 - Baseline Performance: The model must outperform a standard U-Net model under the same dataset conditions in order to evaluate if "incorporating advanced multi-scale context can genuinely eliminate these hazardous boundary and slice-transition errors compared to a 2D baseline."
 - Operational constraints: The model must run comfortably on workstation hardware or local GPUs. Given that patients sensitive medical data must be kept on the hospital's private network.
## Model Choice & Course Concepts
 - Dataset Configuration
    The preprocessed OASIS 2D brain dataset (located on rangpur) consists of pre-segmented groups of train, validate, and test datasets. This separation enforces strict air gap separation to prevent critical data leakage and over-fitting.
 - The Oasis Brain 2D dataset consists of 2D MRI brain slices of several patients.
 - Model Architecture: The chosen architecture is a 2D Context Aggregation Network (CAN) (both normal/easy tiers). Instead of downsampling or pooling, it gains wider, multi-scale contextual information by systematically expanding the receptive field. This enables more context to be captured by the network, enabling greater precision on the pixel level, making it ideal for identifying fine details.

## Preliminary Feasibility evidence
 - Empirical Validation: Initial exploratory prototyping has already been successfully executed, with a working 2D CAN model trained and evaluated to achieve a Dice score exceeding 90% (>0.90).
 - Current Phase: Having verified that the architecture is viable and capable of meeting performance targets, the next immediate phase is implementing a standard 2D U-Net baseline. This will allow for a direct comparative analysis to answer the core course research dilemma: whether multi-scale context aggregation genuinely improves local boundary precision compared to a standard baseline.
 - After determining the performance of the 2D CAN model relative to the 2D U-Net baseline, the next step will be to develop another 2D CAN with a more complex dataset (HipMRI 2D slices.)
 - Compute Resources: Initial training and prototyping were executed successfully on a local high-end consumer gaming GPU (AMD Radeon 9070 xt) without requiring enterprise-grade hardware, proving the model's lightweight efficiency for the easy/normal tier.

## Risks, Budget & Fallback
 - Clinical Safety: Crucially, there must always be a qualified human in between the model and the patient. Given a Dice score of >90%, it is reasonable to expect the model to be accurate, but in medical imaging, deployment risks are severe. If a model hallucinates boundaries, creates disconnected tissue fragments, or misses a critical atrophy zone, it could lead to faulty diagnostic support or flawed volumetric tracking. Furthermore, over-fitting on training slices can cause catastrophic failure on unseen and unique, out-of-distribution patient scans.
 - Computing Budget: The initial pipeline and easy-tier models were trained locally on consumer hardware. However, scaling up to more complex configurations (such as deeper normal/hard variants or 3D volumes) will leverage the university's Rangpur HPC cluster using slurm.
 - Workaround / Fallback: If dilated convolution blocks introduce training instability, gridding artifacts, or fail to cleanly beat the baseline, the fallback plan is to pivot to an Improved 2D U-Net architecture utilizing standard residual skip connections to secure baseline compliance.