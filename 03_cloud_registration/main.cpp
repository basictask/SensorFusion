#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include "main.h"
#include "happly.h"
#include "nanoflann.hpp"
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace happly;
using namespace nanoflann;

//=================[ Made by Daniel Kuknyo ]=================//

// Parameters set by user
const int max_leaf = 10; // Maximum leaf size for KD-tree search
const int num_result = 1; // Number of results for KNN search
const int icp_iter = 50; // Number of iteration for the ICP algorithm
const double icp_error_t = 0.018; // ICP error threshold
const double tricp_error_t = 0.05; // TR-ICP error threshold
const int tricp_points = 50000; // Number of points used by TR-ICP
const string output_dir = "./results/"; // The folder to save results to
const string ply_header = "./data/ply_header.txt"; // Header to add to all output ply files
const vector<Point3i> colors = {Point3i(174, 4, 33), Point3i(172, 255, 36)}; // RGB Colors for point clouds

// Parameters set by program
unsigned long n_rows = -1;
string cloud_name;
float timenow;

// Types defined for the run
typedef KDTreeEigenMatrixAdaptor<MatrixXd> kd_tree;

int main(int argc, char** argv)
{
    // Read point cloud files
    vector<Point3d> vector_1 = read_pointcloud(argv[1]);
    vector<Point3d> vector_2 = read_pointcloud(argv[2]);
    n_rows = min(vector_1.size(), vector_2.size());

    // Assign matrices to the vectors
    MatrixXd cloud_1 = vector2mat(vector_1);
    MatrixXd cloud_2 = vector2mat(vector_2);
    MatrixXd nn = nn_search(cloud_1, cloud_2);
    pair<Matrix3d, Vector3d> transformation = estimate_transformation(cloud_1, cloud_2);

    cout << "=================[ Parameters ]=================" << endl;
    cout << "        KdTree max leaf = " << max_leaf << endl;
    cout << "          NN search dim = " << num_result << endl;
    cout << "         ICP iterations = " << icp_iter << endl;
    cout << "    ICP error threshold = " << icp_error_t << endl;
    cout << "  trICP error threshold = " << tricp_error_t << endl;
    cout << "====================[ Info ]====================" << endl;
    cout << "               Filename = " << cloud_name << endl;
    cout << "      Points in cloud 1 = " << cloud_1.rows() << " x " << cloud_1.cols() << endl;
    cout << "      Points in cloud 2 = " << cloud_2.rows() << " x " << cloud_2.cols() << endl;
    cout << "===============[ Transformation ]===============" << endl;
    cout << "        Rotation matrix = " << endl << transformation.first << endl << endl;
    cout << "     Translation vector = " << endl << transformation.second << endl;
    cout << "     Mean Squared Error = " << mse(cloud_1, cloud_2) << endl;
    cout << "==================== [ ICP ] ===================" << endl;

    // Run the ICP algorithm
    Matrix4d T = icp(cloud_1, cloud_2);
    cout << "Transformation matrix between point clouds: " << endl << T << endl;

    cout << "Done." << endl;
    
    return 0;
}

vector<Point3d> read_pointcloud(char* filename)
{
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
            if(i == 6)
            {
                Point3d tmp = {stof(arr[0]), stof(arr[1]), stof(arr[2])};
                result.push_back(tmp);
            }
        }
    }
    catch (Exception &ex)
    {
        return result;
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

MatrixXd nn_search(const MatrixXd& cloud_1, MatrixXd cloud_2)
{
    MatrixXd nn_result(cloud_1.rows(), 3);
    kd_tree tree_1(3, cref(cloud_1), max_leaf);
    tree_1.index -> buildIndex();
    double total_mse = 0;

    for (int i = 0; i < cloud_1.rows(); i++)
    {
        vector<double> point{cloud_2(i, 0), cloud_2(i, 1), cloud_2(i, 2)};
        unsigned long nn_index = 0; // Index of the closes neighbor will be stored here
        double error = 0; // Error term between the two points
        KNNResultSet<double> result(num_result); // Result of NN search for a single point
        result.init(&nn_index, &error);

        // Run KNN search
        tree_1.index -> findNeighbors(result, &point[0], SearchParams(10));

        // Assign to container matrix
        nn_result(i, 0) = (double)i;
        nn_result(i, 1) = (double)nn_index;
        nn_result(i, 2) = error;

        // Count error
        total_mse += error;
    }

    return nn_result;
}

pair<Matrix3d, Vector3d> estimate_transformation(MatrixXd cloud_1, MatrixXd cloud_2)
{
    // Estimate the translation and rotation between two point clouds cloud_1 and cloud_2
    // The result is a pair object that has the first element as the rotation and the second element as the translation

    // Calculate the centroid for the two point clouds
    Vector3d centroid_1 = cloud_1.colwise().mean();
    Vector3d centroid_2 = cloud_2.colwise().mean();

    // Compute centered points and update both clouds
    cloud_1 = cloud_1.rowwise() - centroid_1.transpose();
    cloud_2 = cloud_2.rowwise() - centroid_2.transpose();

    MatrixXd h = cloud_1.transpose() * cloud_2;

    // Calculate the singular value decomposition for the matrix
    JacobiSVD<MatrixXd> svd(h, ComputeFullU | ComputeFullV);

    MatrixXd u = svd.matrixU();
    MatrixXd v = svd.matrixV();
    MatrixXd v_t = v.transpose();

    Matrix3d R = v_t.transpose() * u.transpose(); // Rotation matrix

    if (R.determinant() < 0 )
    {
        v_t.block<1, 3>(2,0) *= -1;
        R = v_t.transpose() * u.transpose();
    }

    Vector3d t = centroid_1 - R * centroid_2; // Translation vector

    pair<Matrix3d, Vector3d> result(R, t);

    return result;
}

MatrixXd reorder(const MatrixXd& cloud, const MatrixXd& indices)
{
    // Reorder the rows of a matrix to match the ones given by the nearest neighbor search

    MatrixXd result(cloud.rows(), cloud.cols());

    for(int i = 0; i < cloud.rows(); i++)
    {
        int p_0 = (int)indices(i, 0);
        int p_1 = (int)indices(i, 1);
        result.row(p_1) = cloud.row(p_0);
    }

    return result;
}

double mse(const MatrixXd& cloud_1, const MatrixXd& cloud_2)
{
    return (cloud_1 - cloud_2).array().pow(2).sum() / (double)cloud_1.rows();
}

Matrix4d icp(MatrixXd cloud_1, const MatrixXd& cloud_2)
{
    Matrix4d T = Matrix4d::Identity(); // Transformation matrix

    for (int i = 0; i < icp_iter; i++)
    {
        // Compute nearest neighbors
        MatrixXd nn = nn_search(cloud_1, cloud_2);

        // Reorder points
        cloud_1 = reorder(cloud_1, nn);

        // Estimate the transformation between the point clouds
        pair<Matrix3d, Vector3d> transform = estimate_transformation(cloud_1, cloud_2);
        Matrix3d R = transform.first; // Rotation matrix
        Vector3d t = transform.second; // Translation vector

        // Update the transformation matrix
        T.block<3,3>(0,0) = R;
        T.block<3,1>(0,3) = t;

        // Transform the point cloud
        for(int j = 0; j < cloud_1.rows(); j++)
        {
            cloud_1.row(i) = ((R * cloud_1.row(i).transpose()) + t).transpose(); // Apply rotation
        }

        // Compute the mean squared error
        double error = mse(cloud_1, cloud_2);
        cout << "mse=" << error << endl;

        // Check for convergence
        if (error < icp_error_t)
        {
            cout << "              ---ICP Converged!---" << endl;
            cout << "     Mean Squared Error = " << mse(cloud_1, cloud_2) << endl;
            break;
        }
    }

    output_result(cloud_1, cloud_2, "ICP"); // Write the clouds into a file

    return T;
}

void output_result(const MatrixXd& cloud_1, const MatrixXd& cloud_2, const string& method)
{
    // Define the header file
    string line;
    ifstream header_file;
    header_file.open(ply_header);

    // Define the output file
    stringstream out3d;
    out3d << output_dir << cloud_name << "_method=" << method << ".ply";
    ofstream out_file(out3d.str());
    cout << "Printing: " << out3d.str() << "... ";

    if(!header_file.is_open())
    {
        cout << "Error opening header file: " << ply_header << endl;
        exit(EXIT_FAILURE);
    }
    try
    {
        // Output header
        while(getline(header_file, line))
        {
            out_file << line << endl;
        }
        header_file.close();

        // Output clouds
        for(int i = 0; i < cloud_1.rows(); i++)
        {
            // Add first point with first color
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