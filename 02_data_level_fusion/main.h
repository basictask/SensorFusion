#pragma once

#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include <iostream>

cv::Mat ReadImage(char* imname, float scale);

void PrintImage(std::string out_name, cv::Mat image);

cv::Mat CreateGaussianKernel(int window_size);

cv::Mat Filter_Bilateral(const cv::Mat &input);

float CalcMedian(std::vector<float> &v);

cv::Mat Filter_MedianBilateral(const cv::Mat &input);

void PadImages(cv::Mat &input, cv::Mat &guide);

cv::Mat Filter_GuidedBilateral(const cv::Mat &input, const cv::Mat &guide);

cv::Mat NaiveUpsampling(cv::Mat &input, cv::Mat &guide);

cv::Mat IterativeUpsampling(cv::Mat &input, cv::Mat &guide);

void Disparity2PointCloud(std::string out_name, cv::Mat &disp);

cv::Mat GetDifferenceImage(cv::Mat &image, cv::Mat &GT);

void CalcMetrics(cv::Mat &image, cv::Mat &GT, std::string proc_name);

void LogExecution(std::string name);

void GetBestMetric();