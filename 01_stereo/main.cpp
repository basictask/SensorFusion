#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <string> 
#include <fstream>
#include <sstream>
#include "metrics.h"
#include "main.h"

using namespace std;
using namespace cv;

//##-- Made by Daniel Kuknyo --##//

// How to build from source?
// clear && cmake . && make && ./main "data/cones1.png" "data/cones2.png" "output" "data/cones_disp1.png"
// Note: the 4th parameter is optional and refers to the ground truth disparities for the left image

// Starting config for cones
// Naive: window_size=7, dmin=230
// Dynamic: window_size=1. dmin=230, lambda=100

const bool disp_dynamic = true; 				// true --> DP disparities, false --> naive disparities
const bool log_execution = true;				// disable to not log an execution (testing only)
const float imscale = 1.0; 						// resizing factor (for speeding up computation)
const double focal_length = 3740 * imscale; 	// focal length
const double baseline = 160; 					// baseline
const int window_size = 1; 						// size of convolutional mask
const int dmin = 230 * imscale; 				// disparity added due to cropping
const float lambda = 100; 						// weighting parameter (dynamic only)
const float max_disparity = 260; 				// maximal disparity (naive only)
const int d_scale = 1;							// scaling factor for disparities (naive only)

string image_name = "";
string name;

int main(int argc, char** argv)
{
	time_t start, end;
	time(&start); // Timer start

	// Reading images and setting up matrices
	Mat image1 = read_image(argv[1], imscale);
	Mat image2 = read_image(argv[2], imscale);

	int height = image1.size().height;
	int width = image1.size().width;

	Mat disparities = Mat::zeros(height, width, CV_8UC1);
	const std::string output_file = argv[3];

	std::cout << "------------------ Parameters -------------------" << std::endl;
	std::cout << "focal_length = " << focal_length << std::endl;
	std::cout << "baseline = " << baseline << std::endl;
	std::cout << "lambda = " << lambda << std::endl;
	std::cout << "window_size = " << window_size << std::endl;
	std::cout << "minimal disparity = " << dmin << std::endl;
	std::cout << "output filename = " << argv[3] << std::endl;
	std::cout << "input filename = " << image_name << std::endl;
	std::cout << "image size = " << width << "x" << height << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;

	// Compute disparity map
	if (disp_dynamic)
	{
		name = "dynamic";
		StereoEstimation_DP(height, width, image1, image2, disparities); // compute dynamic disparities
	}
	else
	{
		name = "naive";
		StereoEstimation_Naive(height, width, image1, image2, disparities); // compute naive disparities
	}

	// 3D reconstruction
	Disparity2PointCloud(output_file, height, width, disparities); // Create a depth map using triangulation

	// Timer stop
	time(&end);
	double timenow = double(end - start);

	// If there's a 4th argument run diagnostics
	float ssd_mse, ssim, ncc = 0;
	if(argc == 5)
	{
		Mat GT = read_image(argv[4], imscale); 
		
		ssd_mse =  CompareGtSSD(disparities, GT); 
		ssim = CompareGtSSIM(disparities, GT, window_size);
		ncc = CompareGTNCC(disparities, GT);

		cout << "--------------- Metrics Calculated --------------" << endl; 
		cout << "SSD/MSE: " << ssd_mse << endl;
		cout << "   SSIM: " << ssim << endl;
		cout << "    NCC: " << ncc << endl;
		cout << "   Time: " << timenow << "s" << endl;
		cout << "-------------------------------------------------" << endl;
	}

	if(!log_execution)
		return 0;

	// Logging code exection into csv
	fstream fout;
	fout.open("runlogs.csv", ios::out | ios::app);

	fout << output_file << 
	";" << name << 
	";" << image_name << 
	";" << lambda << 
	";" << window_size << 
	";" << dmin << 
	";" << imscale << 
	";" << timenow <<
	";" << ssd_mse << 
	";" << ssim << 
	";" << ncc << "\n";

	cout << "Done logging results." << endl;

	// Write to point cloud and display
	std::stringstream out1;
	out1 << output_file <<
    "_" << name <<
    "_" << image_name <<
    "_lambda=" << lambda <<
    "_kernel=" << window_size <<
    "_dmin=" << dmin <<
    "_scale=" << imscale << ".png";
	cv::imwrite(out1.str(), disparities);
	cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
	cv::imshow(name, disparities);
	cv::waitKey(0);

	return 0;
}

cv::Mat read_image(char* imname, float scale)
{
	// Sets the image_name variable to the name of the image
	string filename = std::string(imname);
	const size_t last_slash_idx = filename.find_last_of("\\/");
	if (std::string::npos != last_slash_idx)
	{
		filename.erase(0, last_slash_idx + 1);
	}
	const size_t period_idx = filename.rfind('.');
	if (std::string::npos != period_idx)
	{
		filename.erase(period_idx);
	}
	if(image_name == "")
	{
		image_name = filename;
	}
	
	// Reads an image received through the params
	Mat image = imread(imname, IMREAD_GRAYSCALE);
	if (!image.data) { 
		throw invalid_argument("No image data for " + (string)(imname));
	}
	if (scale < 1.)
	{
		Mat image_resize;
		resize(image, image_resize, cv::Size(), scale, scale);
		return image_resize;
	}
	return image;
}

void StereoEstimation_Naive(int height, int width, cv::Mat& image1, cv::Mat& image2, cv::Mat& naive_disparities)
{
	int half_window_size = window_size / 2;
#pragma omp parallel for
	for (int i = half_window_size; i < height - half_window_size; ++i) 
	{
		for (int j = half_window_size; j < width - half_window_size; ++j) 
		{
			int min_ssd = INT_MAX;
			int disparity = 0;
			for (int d = -j + half_window_size; d < width - j - half_window_size; ++d) 
			{
				int ssd = 0;
				for (int k = -half_window_size; k < half_window_size + 1; k++)
				{
					for (int l = -half_window_size; l < half_window_size + 1; l++)
					{
						int image1_coord = image1.at<uchar>(i + k, j + l);
						int image2_coord = image2.at<uchar>(i + k, j + l + d);
						ssd += (image1_coord - image2_coord) * (image1_coord - image2_coord);
					}
				}
				if (ssd < min_ssd) 
				{
					min_ssd = ssd;
					disparity = abs(d) * d_scale;
				}
			}
			if (abs(disparity) < max_disparity) 
			{
				naive_disparities.at<uchar>(i - half_window_size, j - half_window_size) = disparity;
			}
#pragma omp critical
			cout << i - half_window_size + 1 << "/" << height - window_size + 1 << "\r" << flush; // Progress
		}
	}
}

void StereoEstimation_DP(int height, int width, cv::Mat& image1, cv::Mat& image2, cv::Mat& dp_disparities)
{
	Size imageSize = image1.size();
	Mat disparityMap = Mat::zeros(imageSize, CV_16UC1);

#pragma omp parallel for
	for (int y_0 = window_size; y_0 < imageSize.height - window_size; ++y_0) 
	{
		Mat C = Mat::zeros(Size(imageSize.width - 2 * window_size, imageSize.width - 2 * window_size), CV_16UC1);
		Mat M = Mat::zeros(Size(imageSize.width - 2 * window_size, imageSize.width - 2 * window_size), CV_8UC1);
		C.at<unsigned short>(0, 0) = 0;
		M.at<unsigned char>(0, 0) = 0;
		for (int i = 1; i < C.size().height; i++)
		{
			C.at<unsigned short>(i, 0) = i * lambda;
			M.at<unsigned char>(i, 0) = 1;
		}
		for (int j = 1; j < C.size().width; j++) 
		{
			C.at<unsigned short>(0, j) = j * lambda;
			M.at<unsigned char>(0, j) = 2;
		}
		for (int r = 1; r < C.size().height; r++)
		{
			for (int l = 1; l < C.size().width; l++)
			{
				Mat window_left = image1(Rect(l, y_0 - window_size, 2 * window_size + 1, 2 * window_size + 1));
				Mat window_right = image2(Rect(r, y_0 - window_size, 2 * window_size + 1, 2 * window_size + 1));
				Mat diff;
				absdiff(window_left, window_right, diff);

                int SAD = sum(diff)[0];
				int c_m = C.at<unsigned short>(r - 1, l - 1) + SAD;
				int c_l = C.at<unsigned short>(r - 1, l) + lambda;
				int c_r = C.at<unsigned short>(r, l - 1) + lambda;

				// Minimizing cost
				int c = c_m;
				int m = 0;
				if (c_l < c) // Occluded from left
				{
					c = c_l;
					m = 1;
					if (c_r < c) // Occluded from right
					{
						c = c_r;
						m = 2;
					}
				}
				C.at<unsigned short>(r, l) = c;
				M.at<unsigned char>(r, l) = m;
			}
		}
		// Create disparity map
		int i = M.size().height - 1;
		int j = M.size().width - 1;
		while (j > 0)
        {
			if (M.at<unsigned char>(i, j) == 0)
			{
				disparityMap.at<unsigned short>(y_0, j) = abs(i - j);
				i--;
				j--;
			}
			else if (M.at<unsigned char>(i, j) == 1) 
			{
				i--;
			}
			else if (M.at<unsigned char>(i, j) == 2)
			{
				disparityMap.at<unsigned short>(y_0, j) = 0;
				j--;
			}
		}
#pragma omp critical
		cout << y_0 - window_size + 1 << "/" << imageSize.height - 2 * window_size << "\r" << flush; // Progress
	}
	Mat disparityMap_CV_8UC1;
	disparityMap.convertTo(disparityMap_CV_8UC1, CV_8UC1);
	dp_disparities = disparityMap_CV_8UC1;
}

void Disparity2PointCloud(const std::string& file, int height, int width, cv::Mat& disp)
{
	std::stringstream out3d;
	out3d << file << "_" << name <<
    "_" << image_name <<
    "_lambda=" << lambda <<
    "_kernel=" << window_size <<
    "_dmin=" << dmin <<
    "_scale=" << imscale << ".xyz";
	std::ofstream outfile(out3d.str());

	float x, y, z;

	cv::Mat disp_tmp;
	disp.convertTo(disp_tmp, CV_32FC1);
	disp_tmp = disp_tmp + dmin;

	for (int u = 0; u < disp.cols; ++u)
	{
		for (int v = 0; v < disp.rows; ++v)
		{
			float d = disp_tmp.at<float>(v, u);
			if (d != 0)
			{
				float u1 = (float)u - (float)width / 2.0;
				float u2 = (float)u + d - (float)width / 2.0;

				float v1 = (float)v - (float)height / 2.0;
				float v2 = v1;

				x = -(baseline * (u1 + u2) / (2 * d));
				y = baseline * v2 / d;
				z = baseline * focal_length / d;
			}
			outfile << x << " " << y << " " << z << std::endl;
		}
	}
	std::cout << "Done witing point cloud." << std::endl;
}