//
// Created by daniel on 2022.12.11..
//
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "happly.h"
#include "nanoflann.hpp"
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace nanoflann;
using namespace happly;
using namespace cv;

vector<Point3d> read_pointcloud(char* filename);

MatrixXd vector2mat(vector<Point3d> vec);