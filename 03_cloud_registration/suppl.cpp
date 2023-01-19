//=================[ Made by Daniel Kuknyo ]=================//
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "main.h"
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>
#include <pcl/registration/icp.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using namespace cv;
using namespace pcl;
using namespace std;
using namespace Eigen;
using namespace happly;
using namespace nanoflann;

//----------[ Parameters set by program ]----------
extern unsigned long n_rows; // NOLINT
extern string cloud_name; // NOLINT
extern double xi; // NOLINT
extern float timenow; // NOLINT

//========[ Supplementary functions for ICP, TR-ICP ]========//

vector<Point3d> read_pointcloud(char* filename)
{
    // Reads a point cloud defined in a char sequence received as a parameter
    // Note: the ply file must contain a header or the container will be empty

    // Sets the cloud_name variable to the name of the point cloud
    // This is just for outputting the correct variable name
    string temp = string(filename);
    const size_t last_slash_idx = temp.find_last_of("\\/");
    if (string::npos != last_slash_idx)
    {
        temp.erase(0, last_slash_idx + 1);
    }
    const size_t period_idx = temp.rfind('.');
    if (string::npos != period_idx)
    {
        temp.erase(period_idx);
    }
    if(cloud_name.empty()) // Only set it on the first iteration
    {
        cloud_name.assign(temp);
    }
    // Reads a ply file of <header> px py pz nx ny nz into a vector of 3D points
    // There must be a header ending with end_header
    // The first 3 values in a line must be the x y z coordinate
    bool read_flag = false;
    string line;
    ifstream myfile;
    myfile.open(filename);
    vector<Point3d> result;
    if(!myfile.is_open())
    {
        cout << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    try
    {
        while(getline(myfile, line)) // Iterate over the lines in the ply file
        {
            if(read_flag) // If the header is passed
            {
                string arr[3];
                int i = 0;
                stringstream ssin(line); // Create a stringstream from line
                while (ssin.good() && i < 3) // Iterate over tokens in the line
                {
                    ssin >> arr[i];
                    i++;
                }
                if(i == 3) // Only add if there's 3 coordinates
                {
                    Point3d tmp = {stof(arr[0]), stof(arr[1]), stof(arr[2])}; // Create and add point to the vector
                    result.push_back(tmp);
                }
            }
            if(line.find("end_header") != string::npos) // If header ended set flag
            {
                read_flag = true;
            }
        }
    }
    catch (Exception &ex)
    {
        cout << "Error while reading data." << endl;
        return result;
    }
    if(result.empty())
    {
        cout << "Error reading ply file. Header not found." << endl;
    }
    return result;
}

MatrixXd vector2mat(vector<Point3d> vec)
{
    // Converts a vector of 3D (double) points into an Eigen matrix
    // We need this conversion because Eigen matrices are of fixed size and point clouds aren't
    MatrixXd result(n_rows, 3);
    for(int i = 0; i < n_rows; i++)
    {
        result(i, 0) = vec[i].x;
        result(i, 1) = vec[i].y;
        result(i, 2) = vec[i].z;
    }
    return result;
}

double obj_func(double x, MatrixXd nn)
{
    // Objective function for golden section search.
    // e(xi) / e^(1+lambda); lambda=2
    int trimmed_len = (int)(x * (double)nn.rows()); // Get the number of rows to calculate mean for
    return nn.block(0, 0, trimmed_len, nn.cols()).col(2).mean() / pow(x, 1 + lambda); // Calculate objective function
}

double golden_section_search(double a, double b, const double& eps, const MatrixXd& nn)
{
    // Golden section search to find optimal overlap parameter
    // Default sectioning is [0.1, 0.9]
    // a: minimum start point, b: maximum end point, eps: tolerance, nn: array for objective function
    double x1 = b - (b - a) / phi;
    double x2 = a + (b - a) / phi;
    double fx1 = obj_func(x1, nn);
    double fx2 = obj_func(x2, nn);
    while (abs(b - a) > eps)
    {
        if (fx1 < fx2)
        {
            b = x2;
            x2 = x1;
            fx2 = fx1;
            x1 = b - (b - a) / phi;
            fx1 = obj_func(x1, nn);
        }
        else
        {
            a = x1;
            x1 = x2;
            fx1 = fx2;
            x2 = a + (b - a) / phi;
            fx2 = obj_func(x2, nn);
        }
    }
    double result = (a + b) / 2;
    cout << "xi=" << result << "; ";
    return result;
}

void output_clouds(const MatrixXd& cloud_1, const MatrixXd& cloud_2, const string& method)
{
    // Outputs two point clouds into a single ply file with two colors defined in the parameter section
    // The ply header is defined in the ./data/ folder and can be customized
    // The number of element vertices gets calculated dynamically: n_rows*2

    // Define the header file
    string line;
    ifstream header_file;
    header_file.open(ply_header);

    // Define the output file
    stringstream out3d;
    out3d << output_dir << cloud_name << "_method=" << method;
    if(apply_init_transformation) // NOLINT
    {
        out3d << "_noise=" << lvl_noise << "_rotation=" << lvl_rotation << "_translation=" << lvl_translation;
    }
    out3d << ".ply";
    ofstream out_file(out3d.str());
    cout << "Printing: " << out3d.str() << "... ";

    if(!header_file.is_open())
    {
        cout << "Error opening header file: " << ply_header << endl;
        exit(EXIT_FAILURE);
    }
    try
    {
        // Output header to the beginning of the ply
        while(getline(header_file, line))
        {
            out_file << line << endl;
            if(line.find("ascii") != string::npos) // Element vertex property -> n_rows*2 for the 2 clouds
            {
                out_file << "element vertex " << (n_rows * 2) << endl;
            }
        }
        header_file.close();
        // Output points in both clouds
        for(int i = 0; i < n_rows; i++)
        {
            // Add second point with second color
            out_file << cloud_1(i, 0) << " " << cloud_1(i, 1) << " " << cloud_1(i, 2) << " ";
            out_file << colors[0].x << " " << colors[0].y << " " << colors[0].z << endl;

            // Add second point with second color
            out_file << cloud_2(i, 0) << " " << cloud_2(i, 1) << " " << cloud_2(i, 2) << " ";
            out_file << colors[1].x << " " << colors[1].y << " " << colors[1].z << endl;
        }
        out_file.close();
    }
    catch (Exception &ex)
    {
        cout << "Error while writing " << out3d.str() << endl;
    }

    cout << "Done." << endl;
}

Matrix4d estimate_T_true(const MatrixXd& cloud_1, const MatrixXd& cloud_2)
{
    // Estimate the true alignment transformation betweeen the point clouds
    // This is a method only used for comparison to get the T_true variable in main.cpp
    PointCloud<PointXYZ>::Ptr cloud_1_pcl(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr cloud_2_pcl(new PointCloud<PointXYZ>);

    // Create pcl containers
    cloud_1_pcl -> resize(n_rows);
    cloud_2_pcl -> resize(n_rows);
    for (int i = 0; i < n_rows; i++) 
    {
        cloud_1_pcl->points[i].x = (float)cloud_1(i, 0);
        cloud_1_pcl->points[i].y = (float)cloud_1(i, 1);
        cloud_1_pcl->points[i].z = (float)cloud_1(i, 2);
        cloud_2_pcl->points[i].x = (float)cloud_2(i, 0);
        cloud_2_pcl->points[i].y = (float)cloud_2(i, 1);
        cloud_2_pcl->points[i].z = (float)cloud_2(i, 2);
    }

    // Estimate the rigid body transformation
    IterativeClosestPoint<PointXYZ, PointXYZ> icp;
    icp.setInputSource(cloud_1_pcl);
    icp.setInputTarget(cloud_2_pcl);
    PointCloud<PointXYZ> aligned;
    icp.align(aligned);

    // Get the transformation matrix
    Matrix4f T = icp.getFinalTransformation();
    return T.cast<double>();
}

void log_execution(const MatrixXd& cloud_1, const MatrixXd& cloud_2, const string& method, const double& error,
                   const int& n_iter, const bool& converged, const Matrix4d& T_true, const Matrix4d& T_pred)
{
    // Logs the execution for a given method. The file it gets written to is defined in main.h [log_file].
    // Order of columns: cloud_name;method;error;n_iter;converged;time;noise;rotation;translation;T_true;T_pred
    // Rotation, translation and noise only get a non-0 value if apply_init_transformation is true
    fstream fout;
    fout.open(log_file, ios::out | ios::app);

    // Output running params
    fout << cloud_name <<
     ";" << method <<
     ";" << error <<
     ";" << n_iter <<
     ";" << converged <<
     ";" << timenow;

    // Output initial transformation parameters
    if(apply_init_transformation) // NOLINT
    {
        fout << lvl_noise <<
         ";" << lvl_rotation <<
         ";" << lvl_translation;
    }
    else
    {
        fout << 0 << ";" << 0 << ";" << 0;
    }

    // Output true and predicted transformation matrices
    fout << ";" << print_mat(T_true, ",") <<
            ";" << print_mat(T_pred, ",") << endl;

    fout.close();
    cout << "Done logging results for " << method << "." << endl;
}

string print_mat(const MatrixXd& mat, const string& sep)
{
    // Prints the contents of a matrix into a single-line string (logging purposes)
    // E.g. the 3x3 identity matrix with sep="," --> "1,0,0,0,1,0,0,0,1"
    stringstream result;
    for(int i = 0; i < mat.rows(); i++)
    {
        for(int j = 0; j < mat.cols(); j++)
        {
            if(i == 0 && j == 0) // Fencepost value
            {
                result << mat(i, j);
            }
            else
            {
                result << sep << mat(i, j); // Print separation character and value
            }
        }
    }
    return result.str();
}