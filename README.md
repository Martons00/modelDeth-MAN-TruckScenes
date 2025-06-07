# Introduction to ModelDepthMAN: Comparative Evaluation of ScaleBiasModel for Depth Estimation

## Context and Motivation
This study introduces a complete pipeline for evaluating depth estimation and transformation models, applied to the MAN TruckScenes mini dataset, with a particular focus on the comparative analysis of three ScaleBiasModel variants characterized by different levels of computational complexity.

## ScaleBiasModel Architectures Investigated
The experiment was designed to systematically compare three distinct ScaleBiasModel configurations, following established principles of neural model scaling. The three variants analyzed are:

- **Light ScaleBiasModel**: Architecture optimized for computational efficiency, with reduced parameters for real-time applications
- **Moderate ScaleBiasModel**: Balanced configuration between performance and computational cost
- **Heavy ScaleBiasModel**: High-capacity architecture to maximize predictive accuracy

This multi-scale comparison strategy allows for analyzing the fundamental trade-off between model complexity and performance, following the neural scaling laws that govern the relationship between model size and predictive capacity.

## Complete Experimental Pipeline
The flow described in the notebook implements an integrated system that combines deep learning techniques, classical regression, and computer vision for a comprehensive evaluation of depth estimation performance.

### Acquisition and Preparation of the MAN TruckScenes Dataset
The pipeline begins with the acquisition of the MAN TruckScenes mini dataset, specifically designed for commercial transport scenarios. RGB images and corresponding LiDAR point clouds are systematically loaded and preprocessed to ensure consistency between training and testing sets. The reduction of the test set is implemented to optimize computational efficiency without compromising the statistical validity of the results.

### Relative Depth Estimation with Depth-Anything V2
The system uses the pre-trained Depth-Anything V2 transformer model, one of the most advanced architectures for monocular depth estimation. This model, trained on 595K labeled synthetic images and over 62M unlabeled real images, generates high-resolution relative depth heatmaps for each image in the training and testing datasets.

### Extraction of Affine Parameters through Polynomial Regression
To establish the relationship between relative and absolute depth, the system implements both linear and quadratic polynomial regression techniques. This approach allows for extracting optimal affine transformation parameters for each sample, providing a mathematically rigorous baseline for comparison with neural models.

### Definition and Training of the Three ScaleBiasHead Variants
The three configurations of the custom neural network (ScaleBiasHead) are defined with scalable architectures that follow compound scaling principles. Each variant takes RGB images and relative depth maps as input, predicting affine transformation parameters through combined loss functions that incorporate depth and parameter errors.

### Multi-Metric Comparative Evaluation
The trained models undergo systematic evaluation on the test set, compared against polynomial regression baselines (2D and 3D polyfit) under various operating conditions. Metrics include Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and depth consistency measures, both globally and under specific lighting conditions.

### Validation with Object Detection and Tangible Metrics
The integration of YOLO for object detection allows for a granular evaluation of performance. For each detected object, the depth predictions of the three ScaleBiasModel models are systematically compared with the LiDAR ground truth, generating detailed error metrics per bounding box. This methodology makes the evaluation particularly relevant for real-world autonomous driving applications.

### Detailed Per-Image Visual Analysis
The pipeline culminates in an in-depth analysis of randomly selected test images. The visualization includes depth map overlays, point cloud data, and quantitative comparisons between the values predicted by the three ScaleBiasModel variants and the polynomial baselines within each detected bounding box.

## Summary of the Architectures of the Three ScaleBiasHead Models

The three ScaleBiasHead models feature architectures designed to balance computational efficiency and accuracy in depth estimation:

- **Light_ScaleBiasHead**: Minimalist structure, uses few channels (from 4 to 32), efficiency-optimized blocks (LightSEBlock, two-branch SimplifiedMultiScale, EfficientDepthwiseConv), and a simplified fully-connected head that maps directly from 32 to 3 output parameters.

- **Moderate_ScaleBiasHead**: Intermediate approach, employs sequential convolutional blocks with larger kernels (5×5 and 3×3), two stages of max pooling, feature map reduction to 4×4 via AdaptiveAvgPool2d, and a more robust fully-connected head (128 hidden neurons, 0.5 dropout).

- **Heavy_ScaleBiasHead**: Most complex architecture, includes advanced multi-scale fusion with four parallel branches, standard SEBlock, DepthwiseSeparableConv with separate normalization, and a two-stage fully-connected head (64→32→3) with dropout to maximize representational capacity.

## MAN TruckScenes Mini
The MAN TruckScenes Mini dataset offers rich multimodal annotations from a comprehensive sensor suite specifically designed for autonomous trucking. Each scene includes synchronized data from 4 RGB cameras, 6 lidar sensors, and 6 radar sensors, providing dense spatial coverage and robust perception in diverse conditions. The lidar sensors deliver high-resolution 3D point clouds, while the radar sensors—featuring 4D capability—capture both range-azimuth and elevation information, yielding around 2,600 points per sample with nearly 360-degree coverage. Additionally, the dataset incorporates precise vehicle state and localization data from two IMUs and a high-precision RTK-GNSS unit. All objects within a range of over 230 meters are manually annotated with 3D bounding boxes, covering 27 object classes and 15 attributes, and are tracked throughout each scene. These detailed annotations support a wide range of perception tasks, including 3D object detection, tracking, and sensor fusion research for autonomous trucking applications.
