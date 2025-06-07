# Introduction to ModelDepthMAN: Comparative Evaluation of ScaleBiasModel for Depth Estimation

## Context and Motivation
This study introduces a complete pipeline for evaluating depth estimation and transformation models, applied to the MAN TruckScenes mini dataset, with a particular focus on the comparative analysis of three ScaleBiasModel variants characterized by different levels of computational complexity.

## ScaleBiasModel Architectures Investigated
The experiment was designed to systematically compare three distinct ScaleBiasModel configurations, following established principles of neural model scaling. 

The three ScaleBiasHead models feature architectures designed to balance computational efficiency and accuracy in depth estimation:

- **Light_ScaleBiasHead**: Minimalist structure, uses few channels (from 4 to 32), efficiency-optimized blocks (LightSEBlock, two-branch SimplifiedMultiScale, EfficientDepthwiseConv), and a simplified fully-connected head that maps directly from 32 to 3 output parameters.

- **Moderate_ScaleBiasHead**: Intermediate approach, employs sequential convolutional blocks with larger kernels (5×5 and 3×3), two stages of max pooling, feature map reduction to 4×4 via AdaptiveAvgPool2d, and a more robust fully-connected head (128 hidden neurons, 0.5 dropout).

- **Heavy_ScaleBiasHead**: Most complex architecture, includes advanced multi-scale fusion with four parallel branches, standard SEBlock, DepthwiseSeparableConv with separate normalization, and a two-stage fully-connected head (64→32→3) with dropout to maximize representational capacity.

The three variants analyzed are:

- **Light ScaleBiasModel**: Architecture optimized for computational efficiency, with reduced parameters for real-time applications
- **Moderate ScaleBiasModel**: Balanced configuration between performance and computational cost
- **Heavy ScaleBiasModel**: High-capacity architecture to maximize predictive accuracy

This multi-scale comparison strategy allows for analyzing the fundamental trade-off between model complexity and performance, following the neural scaling laws that govern the relationship between model size and predictive capacity.

## Complete Experimental Pipeline
The flow described in the notebook implements an integrated system that combines deep learning techniques, classical regression, and computer vision for a comprehensive evaluation of depth estimation performance.

- **Dataset Acquisition and Preparation**  
  Download and preprocess the MAN TruckScenes mini dataset, aligning RGB images and LiDAR point clouds for robust training and testing.

- **Relative Depth Estimation with Depth-Anything V2**  
  Apply the pre-trained Depth-Anything V2 transformer model to generate high-resolution relative depth maps for all images.

- **Affine Parameter Extraction via Polynomial Regression**  
  Use linear and quadratic polynomial regression to derive optimal affine transformation parameters between relative and absolute depths.

- **ScaleBiasHead Model Definition and Training**  
  Define and train three variants of the custom ScaleBiasHead neural network, each predicting affine transformation parameters from RGB and depth inputs using combined loss functions.

- **Multi-Metric Comparative Evaluation**  
  Evaluate trained models against polynomial baselines using RMSE, MAE, and depth consistency, both globally and under different lighting conditions.

- **Object Detection and Tangible Metrics with YOLO**  
  Integrate YOLO object detection to assess model performance per detected object, comparing predicted depths with LiDAR ground truth for each bounding box.

- **Detailed Per-Image Visual Analysis**  
  Visualize selected test images with overlays of depth maps, point clouds, and quantitative comparisons between all model variants and baselines within detected objects.
