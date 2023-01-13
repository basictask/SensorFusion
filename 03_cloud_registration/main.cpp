#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "main.h"
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
const int icp_iter = 30; // Number of iteration for the ICP algorithm
const double icp_error_t = 0.018; // ICP error threshold
const double tricp_error_t = 0.05; // TR-ICP error threshold
const string output_dir = "./results/"; // The folder to save results to
const string ply_header = "./data/ply_header.txt"; // Header to add to all output ply files
const vector<Point3i> colors = {Point3i(174, 4, 33), Point3i(172, 255, 36)}; // RGB Colors for point clouds

// Parameters set by program
unsigned long n_rows = -1;
string cloud_name;
//float timenow;

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
    cout << "Rotation matrix" << endl << transformation.first << endl << endl;
    cout << "Translation vector" << endl << transformation.second << endl << endl;
    cout << "Mean Squared Error = " << calc_error(cloud_1, cloud_2, true) << endl;
    cout << "=====================[ ICP ]====================" << endl;

    // Combine unregistered clouds to see the data before registration
    output_clouds(cloud_1, cloud_2, "before_registration");

    // Run the ICP algorithm
    Matrix4d T = icp(cloud_1, cloud_2);
    cout << "Transformation matrix between point clouds: " << endl << T << endl;

    cout << "Done." << endl;
    
    return 0;
}

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

MatrixXd nn_search(const MatrixXd& cloud_1, MatrixXd cloud_2)
{
    // Nearest-neighbor search: the number of leaves and number of neighbors can be set by hand at the top

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

    MatrixXd covariance = cloud_1.transpose() * cloud_2; // Create 3x3 covariance matrix

    JacobiSVD<Matrix3d> svd(covariance, ComputeFullU | ComputeFullV);

    // Compute the rotation matrix using the U and V matrices from the SVD
    Matrix3d R = svd.matrixU() * svd.matrixV().transpose();

    // Compute the translation vector as the difference between the centroids
    Vector3d t = centroid_2 - R * centroid_1;

    pair<Matrix3d, Vector3d> result(R, t);

    return result;
}

MatrixXd reorder(const MatrixXd& cloud, const MatrixXd& indices)
{
    // Reorder the rows of a matrix to match the ones given by the nearest neighbor search
    MatrixXd result(cloud.rows(), cloud.cols());

    for(int i = 0; i < cloud.rows(); i++)
    {
        int p_0 = (int)indices(i, 0); // Get index for point in cloud 1
        int p_1 = (int)indices(i, 1); // Get index for point in cloud 2
        result.row(p_1) = cloud.row(p_0); // Swap the 2 points
    }

    return result;
}

void transform_cloud(MatrixXd& cloud, const Matrix3d& R, const Vector3d& t)
{
    cloud *= R;
    cloud.rowwise() += t.transpose();
}

double calc_error(const MatrixXd& cloud_1, const MatrixXd& cloud_2, bool mean)
{
    if(mean) // Return the mean squared error between two point clouds
    {
        return (cloud_1 - cloud_2).array().pow(2).sum() / (double)cloud_1.rows();
    }
    else // Return the sum of squared residuals between two point clouds
    {
        return (cloud_1 - cloud_2).array().pow(2).sum();
    }
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
        Matrix4d T_temp = Matrix4d::Identity();
        T_temp.block<3,3>(0,0) = R;
        T_temp.block<3,1>(0,3) = t;
        T *= T_temp;

        transform_cloud(cloud_1, R, t);

        // Compute the mean squared error
        double error = calc_error(cloud_1, cloud_2, true);
        cout << "mse=" << error << endl;

        // Check for convergence
        if (error < icp_error_t)
        {
            cout << "         ------[ ICP Converged! ]------" << endl;
            cout << "                  Error = " << calc_error(cloud_1, cloud_2, true) << endl;
            output_clouds(cloud_1, cloud_2, "ICP"); // Write the clouds into a file
            return T;
        }
    }

    cout << " ------[ ICP Reached max. iterations! ]------" << endl;
    cout << "     Mean Squared Error = " << calc_error(cloud_1, cloud_2, true) << endl;
    output_clouds(cloud_1, cloud_2, "ICP"); // Write the clouds into a file
    return T;
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

