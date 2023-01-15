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

MatrixXd nn_search(const MatrixXd& cloud_1, MatrixXd cloud_2);

pair<Matrix3d, Vector3d> estimate_transformation(MatrixXd cloud_1, MatrixXd cloud_2);

MatrixXd reorder(const MatrixXd& cloud, const MatrixXd& indices);

double calc_error(const MatrixXd& cloud_1, const MatrixXd& cloud_2, bool mean);

void transform_cloud(MatrixXd& cloud, const Matrix3d& R, const Vector3d& t);

Matrix4d icp(MatrixXd cloud_1, const MatrixXd& cloud_2);

MatrixXd sort_matrix(MatrixXd mat);

MatrixXd trim(const MatrixXd& mat, const double& overlap);

void reorder_2(MatrixXd& cloud_1, MatrixXd& cloud_2, const MatrixXd& indices);

Matrix4d tr_icp(MatrixXd cloud_1, MatrixXd cloud_2);

void output_clouds(const MatrixXd& cloud_1, const MatrixXd& cloud_2, const string& method);

double golden_section_search(double a, double b, const double& eps, const MatrixXd& nn);

double obj_func(double x, MatrixXd nn);