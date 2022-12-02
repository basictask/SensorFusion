#pragma once

void StereoEstimation_Naive(int height, int width, cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities);

void Disparity2PointCloud(const std::string& output_file, int height, int width, cv::Mat& disparities);

void StereoEstimation_DP(int height, int width, cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities);

cv::Mat read_image(char* imname, float scale);