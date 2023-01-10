# 3D Sensing and Sensor Fusion
This repository contains projects that implement the processing of LIDAR and photometric data.
## Project Descriptions
1. Stereo: 3D scene reconstruction from two images using the geometric model of standard stereo
	- Naive reconstruction
	- Reconstruction using dynamic programming
3. Data Level Fusion: The following algorithms are implemented:
	- Bilateral filtering
	- Median bilateral filtering
	- Guided bilateral filtering
	- Naive bilateral upsampling
	- Iterative bilateral upsampling
	- 3D reconstruction using disparities obtained from upsampling
4.  Cloud registration: 3D point cloud processing using: 
	- Iterative closest point
	- Trimmed iterative closest point
## Methodology
Each folder contains the source data, the uncompiled source codes and a study with metrics and results. The different algorithms are analyzed and the aspects are displayed with statistics regarding accuracy, loss, processing times etc...
