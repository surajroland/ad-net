# ADNet Depth Estimation: Complete Technical Analysis

**ADNet definitively includes sophisticated depth estimation capabilities** integrated as a core component of its architecture. The model employs a multi-faceted approach to depth prediction that evolved significantly from earlier versions, with both auxiliary supervision and architectural innovations specifically designed to address camera-based 3D perception challenges.

## Dense depth supervision drives v3's core architecture

ADNet implements a **dense depth branch** as auxiliary supervision during training, representing a major architectural component inherited and refined from v2. The depth estimation system uses LiDAR point clouds for ground truth supervision, operating through a dedicated 3-layer network with 256 embedding dimensions and 0.2 loss weight. This dense depth supervision serves dual purposes: stabilizing training convergence when using ImageNet pretraining instead of FCOS3D, and preventing gradient collapse issues in early training stages.

The technical implementation processes multiple Feature Pyramid Network (FPN) levels with input channels [256, 512, 1024, 2048] from the backbone, producing multi-scale depth estimates. During training, the system applies vanilla L1 loss for depth supervision, while the depth branch is deactivated during inference for computational efficiency. The depth supervision strategy uses PNG/HDF5 format depth maps at 3×1080×1920 resolution, with LiDAR point clouds providing auxiliary supervision throughout the training process.

## Instance-level depth reweighting addresses fundamental 3D projection challenges  

Beyond dense supervision, ADNet inherits the **instance-level depth reweight module** first introduced in v1. This sophisticated mechanism alleviates ill-posed issues in 3D-to-2D projection by reweighting instance features using depth confidence sampled from predicted depth distributions. The module integrates seamlessly into the deformable aggregation process, where 13 keypoints (7 fixed + 6 learnable) sample features across multiple views, scales, and timestamps.

The depth reweighting operates through 4D sampling architecture using keypoints projected to multi-view/scale/timestamp image features, followed by hierarchical feature fusion. Instance features receive depth confidence weights during the **Efficient Deformable Aggregation (EDA)** process, which combines bilinear grid sampling with weighted summation in a custom CUDA operation. This approach reduces GPU memory usage by 51% while improving inference speed by 42%, demonstrating how depth estimation integrates efficiently into the overall architecture.

## Architectural evolution shows sophisticated depth estimation progression

The development of depth capabilities across Sparse4D versions reveals strategic evolution:

**Sparse4D v1** introduced instance-level depth reweight as a self-supervised solution, training without additional LiDAR supervision while addressing 3D-to-2D projection uncertainties. **Sparse4D v2** added revolutionary dense depth supervision using LiDAR point clouds, significantly improving convergence and accuracy. **ADNet** consolidates and optimizes these capabilities, integrating depth estimation with end-to-end tracking while maintaining all depth components from previous versions.

The v3 architecture enhances depth estimation through quality estimation heads that incorporate spatial positioning metrics: **centerness** (`C = exp(-‖[x,y,z]pred - [x,y,z]gt‖2)`) and **yawness** (`Y = [sin yaw, cos yaw]pred · [sin yaw, cos yaw]gt`). These metrics improve depth-related translation error (mATE) by 2.8%, demonstrating measurable improvements in spatial accuracy that directly benefits depth estimation performance.

## Complete prediction head architecture includes depth components

ADNet implements five primary prediction heads working in concert:

The **classification head** uses focal loss (`gamma: 2.0`, `alpha: 0.25`, `loss_weight: 2.0`) for object detection. The **regression head** outputs 10-dimensional boxes including 3D coordinates and velocity vectors with specialized loss weighting. The **quality estimation heads** (new in v3) evaluate spatial alignment through centerness and yawness metrics. The **instance ID head** enables tracking through cross-entropy with label smoothing. Finally, the **dense depth branch** (`embed_dims: 256`, `num_depth_layers: 3`, `loss_weight: 0.2`) provides auxiliary depth supervision.

The decoupled attention mechanism in v3 further enhances depth estimation by replacing addition with concatenation in attention calculations, reducing feature interference between anchor embeddings and instance features. This architectural improvement prevents outlier attention weights that could negatively impact depth estimation accuracy.

## Official implementation confirms comprehensive depth integration

The official Horizon Robotics implementation at `github.com/HorizonRobotics/Sparse4D` contains complete depth estimation modules built on the MMDetection3D framework. The repository structure includes depth components in `projects/mmdet3d_plugin/models/dense_heads/` with configuration files specifying exact depth branch parameters.

The implementation demonstrates production readiness through deployment on Horizon's Journey 5 computing platform and integration with NVIDIA TAO for optimized inference. Training requires 8 RTX 3090 GPUs with 24GB memory, achieving 19.8 FPS inference speed with ResNet50 backbone. Performance metrics show significant depth-related improvements: mean Average Translation Error (mATE) of 0.553 in v3 compared to 0.598 in v2, with overall 71.9% NDS and 67.7% AMOTA on nuScenes test set.

## Conclusion: Sophisticated depth estimation as core architectural component

ADNet represents a mature implementation of camera-based depth estimation, integrating multiple complementary approaches: dense LiDAR-supervised depth prediction, instance-level depth reweighting, and quality estimation metrics. The architecture demonstrates how depth estimation can be efficiently integrated into sparse 3D perception frameworks while maintaining real-time performance requirements for autonomous driving applications. The official implementation provides complete access to these depth estimation capabilities, confirming Horizon Robotics' commitment to production-ready depth-aware 3D perception systems.