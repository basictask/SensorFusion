# Stereo Estimation
The repository contains a C++ implementation for the following problems: 
- Estimating a disparity map using the naive method
- Stereo estimation using Dynamic Programming
- 3D Scene reconstruction using the estimated disparity maps
Below is a short explanation of the folder structure of the repository.
### Data
Source files (stereo images and disparities) for the reconstruction problem.
### Doc 
Documentation in .pdf and .lyx format. The best parameters for each approach are also here in a .csv dump.
### Oriented_clouds
The point clouds that have been generated have also been rendered into a .ply format. These point clouds can be found in this folder.
### Plots
All the diagrams for the statistical analysis.
### Results
The disparity maps and the corresponding stereo reconstruction.
## Main folder
### main.cpp
The stereo reconstruction and disparity estimation happens here.
### metrics.cpp
The metrics used to evaluate the point clouds are defined here:
- SSD
- NCC 
- SSIM
### plotmaker.py
The script used to create plots. The input to this file is runlogs.csv where all the running parameters have been logged.
### surfNormalEstimator.py
The script used to render an .xyz point cloud to an .ply point cloud by estimating the normal vector to each point. 
