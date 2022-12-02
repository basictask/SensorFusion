#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include "main.h"
#include "metrics.h"

using namespace cv;
using namespace std;

// Build from source 
// Args: [1]image_to_filter [2]image_to_upsample
// clear && cmake . && make && ./main "data/aloe1.png" "data/aloe_disp1.png"
// data: aloe, art, baby, cloth, cones, flowerpots, lampshade, midd, monopoly, plastic, rocks, wood

// Parameters set by user
int window_size = -1;									// kernel size for all filtering operations
vector<int> windows_to_try = {3, 5, 7, 11};		 		// iterate over there window sizes for optimization
float sigma = 5;										// distribution parameter for the gaussian curve
const float scale = 1.0; 								// scaling factor for reading images
const bool show_images = false;							// show result images in modal windows
const bool print_images = true;							// print the resulting images
const double baseline = 160;							// baseline of camera
const int dmin = 200 * scale;							// minimum disparity for 3D reconstruction
const double focal_length = 3740 * scale;				// focal length of camera
const string output_dir = "./results/";					// target directory for output images
const string param_file = "bestparams_win.csv";			// best performing params get printed here

// Parameters set by program
string image_name = "";
float ssd_mse = 0;
float ssim = 0;
float ncc = 0;
float timenow;
vector<float> metriclogs;

int main(int argc, char** argv) 
{
	time_t start, end;

	for(int idx = 0; idx < windows_to_try.size(); idx++) // Iterate over given window sizes
	{
		cout << "\033[2J\033[1;1H"; 				// Clear screen
		
		// Setup matrices
		Mat im = ReadImage(argv[1], scale);			// Read the color image image
		Mat guide = ReadImage(argv[1], scale); 		// The guide image can be set freely here
		Mat depth_fs = ReadImage(argv[2], scale); 	// Fullscale depth image
		Mat depth = ReadImage(argv[2], 0.5*scale); 	// The ground truth disparity image belonging to argv[1]
		window_size = windows_to_try.at(idx);  		// Set the window size for the current iteration  
		
		cout << "------------------ Parameters -------------------" << endl;
		cout << "              progress = " << (float)idx / windows_to_try.size() * 100 << "%" << endl;
		cout << "            image name = " << image_name << endl; 
		cout << "            image size = " << im.cols << "*" << im.rows << endl;
		cout << "            depth size = " << depth.cols << "*" << depth.rows << endl;
		cout << "                 scale = " << scale << endl;
		cout << "                kernel = " << window_size << endl;
		cout << "                 sigma = " << sigma << endl;
		cout << "           kernel list = [" << windows_to_try.at(0);
		for(int i = 1; i < windows_to_try.size(); i++)
			cout << ", " << windows_to_try.at(i); 
		cout << "]" << endl;
		cout << "              baseline = " << baseline << endl;
		cout << "                  dmin = " << dmin << endl;
		cout << "          focal length = " << focal_length << endl;
		cout << "-------------------------------------------------" << endl;
		
		//---------------------- Filtering -----------------------
		// Bilateral filtering
		Mat output = Filter_Bilateral(im); 
		PrintImage("filter_bilateral", output); 
		
		// Median bilateral filtering
		output = Filter_MedianBilateral(im);
		PrintImage("median_bilateral", output);
		
		// Guided bilateral filtering
		medianBlur(im, guide, window_size); 
		output = Filter_GuidedBilateral(im, guide); 
		PrintImage("guided_bilateral", output);
		
		//-------------- Bilateral naive upsampling --------------
		time(&start);
		output = NaiveUpsampling(depth, guide);
		time(&end);
		timenow = float(end - start);
		PrintImage("upsampling_naive", output);

		// Difference image
		Mat diff = GetDifferenceImage(depth_fs, output);
		PrintImage("difference_naive", diff);

		// Output 3D point cloud from naive upsampling
		Disparity2PointCloud("pointcloud_naive", output);

		// Metrics for naive upsampling
		CalcMetrics(depth_fs, output, "naive");

		// Log execution
		LogExecution("naive");

		//----------- Bilateral iterative upsampling -------------
		time(&start);
		output = IterativeUpsampling(depth, guide);
		time(&end);
		PrintImage("upsampling_iterative", output);

		// Difference image
		diff = GetDifferenceImage(depth_fs, output);
		PrintImage("difference_iterative", diff);

		// Output 3D point cloud from iterative upsampling
		Disparity2PointCloud("pointcloud_iterative", output);

		// Metrics for iterative upsampling
		CalcMetrics(depth_fs, output, "iterative");

		// Log execution
		LogExecution("iterative");
	}

	// Logging the best performing SSIM
	GetBestMetric(); 
	
	cout << "Done." << endl;

	return 0;
}

void PrintImage(string out_name, Mat image)
{
	// Outputs the image with a name set dynamically
	stringstream out1;
	out1 << output_dir << image_name << "_" << out_name << "_size=" << scale << "_kernel=" << window_size << ".png"; 

	if(print_images)
	{
		cout << "Printing: " << out1.str() << endl;
		imwrite(out1.str(), image);
	}
	
	if(show_images)
	{
		namedWindow(out1.str(), WINDOW_AUTOSIZE);
		imshow(out1.str(), image);
		waitKey(0);
	}
}

Mat ReadImage(char* imname, float i_scale)
{
	// Sets the image_name variable to the name of the image
	string filename = string(imname);
	const size_t last_slash_idx = filename.find_last_of("\\/");
	if (string::npos != last_slash_idx)
	{
		filename.erase(0, last_slash_idx + 1);
	}
	const size_t period_idx = filename.rfind('.');
	if (string::npos != period_idx)
	{
		filename.erase(period_idx);
	}
	if(image_name == "")
	{
		image_name.assign(filename);
	}
	
	// Read image
	Mat image = imread(imname, IMREAD_GRAYSCALE);
	
	if (!image.data) 
	{ 
		throw invalid_argument("No image data for " + (string)(imname));
	}
	if (i_scale < 1.) // Resize if the scale param is smaller than 1.
	{
		Mat image_resize;
		resize(image, image_resize, Size(), i_scale, i_scale);
		return image_resize;
	}

	return image;
}

Mat CreateGaussianKernel(int window_size)
{
	Mat kernel(Size(window_size, window_size), CV_32FC1);

	int half_window_size = window_size / 2;

	// see: lecture_03_slides.pdf, Slide 13
	const double k = 2.5;
	const double r_max = sqrt(2.0 * half_window_size * half_window_size);
	const double sigma = r_max / k;

	// sum is for normalization
	float sum = 0.0;

	for (int x = -window_size / 2; x <= window_size / 2; x++)
	{
		for (int y = -window_size / 2; y <= window_size / 2; y++)
		{
			float val = exp(-(x * x + y * y) / (2 * sigma * sigma));
			kernel.at<float>(x + window_size / 2, y + window_size / 2) = val;
			sum += val;
		}
	}
	
	// normalising the Kernel
	for (int i = 0; i < window_size; ++i)
	{
		for (int j = 0; j < window_size; ++j)
		{
			kernel.at<float>(i, j) /= sum;
		}
	}
	
	return kernel;
}

Mat Filter_Bilateral(const Mat &input)
{
	const auto width = input.cols;
	const auto height = input.rows;

	Mat gaussianKernel = CreateGaussianKernel(window_size);

	Mat output = Mat::zeros(input.size(), input.type());

	auto d = [](float a, float b)
	{
		return abs(a - b);
	};

	auto p = [](float val)
	{
		const float sigmaSq = sigma * sigma;
		const float normalization = sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r)
	{
		for (int c = window_size / 2; c < width - window_size / 2; ++c)
		{
			float sum_w = 0;
			float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i)
			{
				for (int j = -window_size / 2; j <= window_size / 2; ++j)
				{
					// Compute the range difference to the center of the mask
					float range_difference = d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));

					// Compute the range kernel's value
					float w = p(range_difference) * gaussianKernel.at<uchar>(i + window_size / 2, j + window_size / 2);

					// Combine the weights
					sum += input.at<uchar>(r + i, c + j) * w;
					sum_w += w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;
		}
	}

	return output;
}

float CalcMedian(vector<float> &v)
{
	// Returns the median of the vector passed as a reference
	// No need to completely sort the container by value
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

Mat Filter_MedianBilateral(const Mat &input)
{
	const auto width = input.cols;
	const auto height = input.rows;

	Mat gaussianKernel = CreateGaussianKernel(window_size);

	Mat output = Mat::zeros(input.size(), input.type());

	auto d = [](float a, float b)
	{
		return abs(a - b);
	};

	auto p = [](float val)
	{
		const float sigmaSq = sigma * sigma;
		const float normalization = sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r)
	{
		for (int c = window_size / 2; c < width - window_size / 2; ++c)
		{
			vector<float> med;
			vector<float> med_w;	

			for (int i = -window_size / 2; i <= window_size / 2; ++i)
			{
				for (int j = -window_size / 2; j <= window_size / 2; ++j)
				{
					// Compute the range difference to the center of the mask
					float range_difference = d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));

					// Compute the range kernel's value
					float w = p(range_difference) * gaussianKernel.at<uchar>(i + window_size / 2, j + window_size / 2);

					// Combine the weights
					med.push_back(input.at<uchar>(r + i, c + j) * w);
					med_w.push_back(w);
				}
			}

			output.at<uchar>(r, c) = CalcMedian(med) / CalcMedian(med_w);
		}
	}
	return output;
}

void PadImages(Mat &input, Mat &guide)
{
	if (input.size().height == guide.size().height && input.size().width == guide.size().width)
	{
		return;
	}
	else if(input.size().height >= guide.size().height && input.size().width >= guide.size().width)
	{
		Mat padded; 
		int rowPadding = input.rows - guide.rows;
		int colPadding = input.cols - guide.cols;
		copyMakeBorder(guide, padded, 0, rowPadding, 0, colPadding, BORDER_CONSTANT, Scalar::all(0));		
		guide = padded;
	}
	else if (input.size().height < guide.size().height && input.size().width < guide.size().width)
	{
		Mat padded; 
		int rowPadding = guide.rows - input.rows;
		int colPadding = guide.cols - input.cols;
		copyMakeBorder(input, padded, 0, rowPadding, 0, colPadding, BORDER_CONSTANT, Scalar::all(0));
		input = padded;
	}
	else
	{
		throw invalid_argument("Guide image has to be either larger or smaller than input image.");
		// I was too lazy to create a padding for all edge cases... This will work for demonstration
	}
}

Mat Filter_GuidedBilateral(const Mat &input, const Mat &guide)
{
	const auto width = input.cols;
	const auto height = input.rows;

	Mat gaussianKernel = CreateGaussianKernel(window_size);

	Mat output = Mat::zeros(input.size(), input.type());

	auto d = [](float a, float b)
	{
		return abs(a - b);
	};

	auto p = [](float val)
	{
		const float sigmaSq = sigma * sigma;
		const float normalization = sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r)
	{
		for (int c = window_size / 2; c < width - window_size / 2; ++c)
		{
			float sum_w = 0;
			float sum = 0;

			for (int i = -window_size / 2; i <= window_size / 2; ++i)
			{
				for (int j = -window_size / 2; j <= window_size / 2; ++j)
				{
					// Compute the range difference to the center of the mask
					float range_difference = d(input.at<uchar>(r, c), input.at<uchar>(r + i, c + j));

					// Compute the range kernel's value
					float w = p(range_difference) * gaussianKernel.at<uchar>(i + window_size / 2, j + window_size / 2);

					// Combine the weights
					sum += guide.at<uchar>(r + i, c + j) * w;
					sum_w += w;
				}
			}

			output.at<uchar>(r, c) = sum / sum_w;
		}
	}
	return output;
}

Mat NaiveUpsampling(Mat &input, Mat &guide)
{
	const auto width = guide.cols;
	const auto height = guide.rows;	

	float width_factor = (float)input.cols / (float)guide.cols;
	float height_factor = (float)input.rows / (float)guide.rows;

	Mat output = Mat::zeros(guide.size(), guide.type());

	Mat gaussianKernel = CreateGaussianKernel(window_size);

	auto d = [](float a, float b)
	{
		return abs(a - b);
	};

	auto p = [](float val)
	{
		const float sigmaSq = sigma * sigma;
		const float normalization = sqrt(2 * M_PI) * sigma;
		return (1 / normalization) * exp(-val / (2 * sigmaSq));
	};

	for (int r = window_size / 2; r < height - window_size / 2; ++r)
	{
		for (int c = window_size / 2; c < width - window_size / 2; ++c)
		{
			float sum = 0;
			float sum_w = 0;
			
			for (int i = -window_size / 2; i <= window_size / 2; ++i)
			{
				for (int j = -window_size / 2; j <= window_size / 2; ++j)
				{
					float range_difference = d(guide.at<uchar>(r, c), guide.at<uchar>(r + i, c + j));

					float w = p(range_difference) * gaussianKernel.at<uchar>(i + window_size / 2, j + window_size / 2);

					float x = round((r + i) * height_factor);
					float y = round((c + j) * width_factor); 

					sum += (float)input.at<uchar>(x, y) * w;
					sum_w += w;
				}
			}
			output.at<uchar>(r, c) = sum / sum_w;			
		}
	}

	return output;		
}

Mat IterativeUpsampling(Mat &D, Mat &I)
{
	int uf = log2(I.size().height / D.size().height);
	Mat D_ = D;

	for(int i = 1; i < uf-1; ++i)
	{
		resize(D_, D_, Size(), 2, 2);
		
		Mat I_lo;
		resize(I, I_lo, D_.size());

		D_ = Filter_GuidedBilateral(I_lo, D_);	
	}

	resize(D_, D_, I.size());
	D_ = Filter_GuidedBilateral(I, D_);
	
	return D_;
}

void Disparity2PointCloud(string out_name, Mat &disp)
{
	stringstream out3d;
	out3d << output_dir << image_name << "_" << out_name << "_size=" << scale << "_kernel=" << window_size << ".xyz";
	ofstream outfile(out3d.str());
	cout << "Printing: " << out3d.str() << endl;

	const auto width = disp.cols;
	const auto height = disp.rows;	

	float x, y, z;

	Mat disp_tmp;
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
			outfile << x << " " << y << " " << z << endl;
		}
	}
	cout << "Printing: done." << endl;
}

Mat GetDifferenceImage(Mat &image, Mat &GT)
{
	const auto width = image.cols;
	const auto height = image.rows;	

	Mat output = Mat::zeros(image.size(), image.type());

	for (int r = 0; r < height; ++r)
	{
		for (int c = 0; c < width; ++c)
		{
			output.at<uchar>(r, c) = abs(image.at<uchar>(r, c) - GT.at<uchar>(r, c));
		}
	}

	return output;
}

void CalcMetrics(Mat &image, Mat &GT, string proc_name)
{
	ssd_mse =  CompareGtSSD(image, GT); 
	ssim = CompareGtSSIM(image, GT, window_size);
	ncc = CompareGTNCC(image, GT);

	cout << "--------------- Metrics Calculated --------------" << endl; 
	cout << "                Method = " << proc_name << endl;
	cout << "               SSD/MSE = " << ssd_mse << endl;
	cout << "                  SSIM = " << ssim << endl;
	cout << "                   NCC = " << ncc << endl;
	cout << "-------------------------------------------------" << endl;

	metriclogs.push_back(ssim);
}

void LogExecution(string name)
{
	fstream fout;
	fout.open("runlogs.csv", ios::out | ios::app);

	fout << name << 
	";" << image_name << 
	";" << window_size << 
	";" << sigma << 
	";" << dmin << 
	";" << scale << 
	";" << timenow <<
	";" << ssd_mse << 
	";" << ssim << 
	";" << ncc << "\n";

	fout.close();

	cout << "Done logging results." << endl;
}

void GetBestMetric()
{
	// Calculates and prints the best performing window size and the corresponding SSIM
	if(metriclogs.size() == 2)
	{
		return;
	}

	float best_metric = numeric_limits<float>::min(); 
	int best_i = -1;
	int best_j = -1;
	for(int i = 0; i < metriclogs.size(); ++i)
	{
		float metric = metriclogs.at(i);
		if(metric > best_metric)
		{
			best_metric = metric;
			best_i = i % 2;
			best_j = i;
		}
	}

	// Every even value in the metriclog is naive, every off is iterative
	// metriclogs -> [naive, iterative, naive, iterative, ...] 
	string best_method;
	if(best_j % 2 == 0)
	{
		best_method = "naive";
	}
	else
	{
		best_method = "iterative";
	}

	cout << "------------- Optimalization metrics ------------" << endl;
	cout << "         Optimal sigma = " << windows_to_try.at(best_i) << endl;
	cout << "          Optimal SSIM = " << best_metric << endl;
	cout << "        Optimal method = " << best_method << endl;
	cout << "-------------------------------------------------" << endl;

	// Write params into file
	fstream fout;
	fout.open(param_file, ios::out | ios::app);
	fout << image_name << ";" <<  windows_to_try.at(best_i) << ";" << best_metric << ";" << best_method << endl;
	fout.close();	
}
