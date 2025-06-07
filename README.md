# Introduction to ModelDepthMAN: Comparative Evaluation of ScaleBiasModel for Depth Estimation

## Context and Motivation
This study introduces a complete pipeline for evaluating depth estimation and transformation models, applied to the MAN TruckScenes mini dataset, with a particular focus on the comparative analysis of three ScaleBiasModel variants characterized by different levels of computational complexity.

For more details, refer to the [Report](Report_modelDepth_martone_s324807.pdf).

All the results and code are available in the [Results](Results) and [Scripts](Scripts) directories, respectively.

[![LidAR Point Cloud](/assets/lidar_trucks.png)](assets/lidar_trucks.png)

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

[![Experimental Pipeline](/assets/flow.png)](assets/experimental_pipeline.png)

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

[![3D depth estimation plot showing MAN truck scene with overlaid predicted depth map, axes labeled with spatial coordinates, and color gradient indicating depth values. The scene includes a truck and surrounding environment, with quantitative depth information visualized. No visible text in the image. Neutral, technical tone.](/Results/result_light/PLOT/02/model_3d_plot_depth_02.png)](/Results/result_light/PLOT/02/model_3d_plot_depth_02.png)
---

## Models Overview Performances

- **Light Model**
  - Best overall mean absolute error: 1.711m
  - Excels in real-time applications and mixed lighting (error as low as 0.787m)
  - Outperforms polynomial regression by 9.35x in favorable conditions
  - Superior with sparse datasets (3.791m error vs. 10.223m for classical approaches)
  - Ideal for commercial deployment due to speed and accuracy

- **Moderate Model**
  - Balanced between complexity and performance
  - Shows greater variability but strong with small datasets (5.961m error vs. 9.530m for polynomial regression)
  - 60% improvement over classical methods
  - Acceptable inference times and higher representational capacity than Light model
  - Suitable for consistent, moderate performance needs

- **Heavy Model**
  - Most sophisticated, achieves near-perfect balance with classical approaches
  - Outperforms classical methods in 50.2% of cases
  - Excels in well-lit conditions (error: 3.24m)
  - 67.9% and 51.4% improvement over polynomial regression for small and medium datasets
  - Adaptive in mixed lighting, outperforming classical in 63.7% of cases

- **Heavy Model with 100 Epochs**
  - Maintains strengths in specific domains, especially with sparse datasets (surpasses polynomial regression by 35.1%)
  - Nearly equivalent to classical in mixed lighting (8.014m vs. 8.204m error)
  - Exceptional in high-density point cloud detections (error as low as 3.73m)
  - Suitable for highly specialized, advanced applications

[![Model Performance Overview](/assets/comparisonMeanError.png)](assets/comparisonMeanError.png)
---

## Analysis

The comparative analysis between ScaleBiasModel neural network variants and polynomial regression reveals clear trade-offs. The Light model stands out for its efficiency, achieving the lowest mean absolute error among neural configurations and excelling in mixed lighting and sparse data scenarios. Its computational speed and predictive accuracy make it ideal for real-time, commercial applications. The Moderate model offers a robust balance, performing well with small datasets and maintaining versatility in complex environments. The Heavy model and its 100-epoch variant demonstrate peak performance under optimal conditions, with strong improvements over classical methods, especially in well-lit environments and with sparse data. However, extended training does not always yield better results, as seen with the Heavy_100_epochs model recording higher errors than the standard Heavy model.

[![Model Performance Analysis](/assets/comparisonSummary.png)](assets/comparisonSummary.png)

Polynomial regression, while less sophisticated, provides remarkable consistency and stability across all scenarios, often outperforming neural models with large datasets or in low-light conditions. Neural networks show greater variability and sensitivity to architecture and training parameters, excelling with small datasets but sometimes struggling with larger ones or challenging lighting.

---

## Implications

The Light model’s low mean absolute error and real-time capabilities make it a strong candidate for commercial trucking, where rapid, accurate depth estimation is essential for safety. Although neural models are sensitive to environmental conditions—especially low-light scenarios—their superior performance with sparse datasets is highly relevant for real-world trucking, where sensor occlusion or weather interference may limit data availability. The varying performance across point cloud densities suggests that with optimized sensor fusion, larger training datasets, and dedicated computational resources, these architectures could become even more robust for autonomous vehicle perception systems.