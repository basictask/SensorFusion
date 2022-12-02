#pragma once
#include <opencv2/opencv.hpp>

float CompareGtSSD(cv::Mat image, cv::Mat GT);

float CompareGTNCC(cv::Mat image, cv::Mat GT);

float CompareGtSSIM(cv::Mat image1, cv::Mat image2, int kernel_size);