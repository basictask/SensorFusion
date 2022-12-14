#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "main.h"
#include "happly.h"
#include "nanoflann.hpp"
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace nanoflann;
using namespace happly;
using namespace cv;

// Made by Daniel Kuknyo

int main(int argc, char** argv)
{
    // Read point cloud files
    MatrixXd cloud_1 = vector2mat(read_pointcloud(argv[1]));
    MatrixXd cloud_2 = vector2mat(read_pointcloud(argv[2]));

    cout << "====================[ Info ]====================" << endl;
    cout << "      Points in cloud 1 = " << cloud_1.rows() << " x " << cloud_1.rows() << endl;
    cout << "      Points in cloud 2 = " << cloud_2.rows() << " x " << cloud_2.cols() << endl;
    cout << "================================================" << endl;



    return 0;
}

vector<Point3d> read_pointcloud(char* filename)
{
    // Reads a ply file of <header> px py pz nx ny nz into a vector of 3D points
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
        while(getline(myfile, line))
        {
            string arr[6];
            int i = 0;
            stringstream ssin(line);
            while (ssin.good() && i < 6)
            {
                ssin >> arr[i];
                i++;
            }
            if(i==6)
            {
                Point3d tmp = {stof(arr[0]), stof(arr[1]), stof(arr[2])};
                result.push_back(tmp);
            }
        }
    }
    catch (Exception ex)
    {
        return result;
    }
    return result;
}

MatrixXd vector2mat(vector<Point3d> vec)
{
    // Converts a vector of points into an Eigen matrix
    // We need this conversion because Eigen matrices are of fixed size and point clouds aren't
    unsigned long n = vec.size();
    MatrixXd result(n, 3);

    for(int i = 0; i < n; i++)
    {
        result(i, 0) = vec[i].x;
        result(i, 1) = vec[i].y;
        result(i, 2) = vec[i].z;
    }

    return result;
}