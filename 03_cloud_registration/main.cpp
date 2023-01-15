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
const int icp_iter = 100; // Number of iteration for the ICP algorithm
const double icp_error_t = 0.7; // ICP error threshold
const int tricp_iter = 100; // Number of iterations for the TR-ICP algorithm
const double tricp_error_t = 0.7; // TR-ICP error threshold
const double tricp_error_change_t = 0.5; // TR-ICP error change threshold
const double phi = (1 + sqrt(5)) / 2; // Golden ratio
const int lambda = 2; // Tolerance parameter for golden section objective function

const string output_dir = "./results/"; // The folder to save results to
const string ply_header = "./data/ply_header.txt"; // Header to add to all output ply files
const vector<Point3i> colors = {Point3i(174, 4, 33), Point3i(172, 255, 36)}; // RGB Colors for point clouds

// Parameters set by program
unsigned long n_rows = -1;
string cloud_name;
double xi; // Maximum overlap parameter for tr-icp. N_po = xi * N_p
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

    // Combine unregistered clouds to see the data before registration
    output_clouds(cloud_1, cloud_2, "before_registration");

    // Run the ICP algorithm
    cout << "=====================[ ICP ]====================" << endl;
//    Matrix4d T_icp = icp(cloud_1, cloud_2);
//    cout << "Transformation matrix between point clouds: " << endl << T_icp << endl;

    // Run the TR-ICP algorithm
    cout << "===================[ TR-ICP ]===================" << endl;
    Matrix4d T_tr_icp = tr_icp(cloud_1, cloud_2);
    cout << "Transformation matrix between point clouds: " << endl << T_tr_icp << endl;

    cout << "====================[ Done ]====================" << endl;
    
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
    // Nearest-neighbor search -> params can be set at the top of the script
    // The number of leaves: num_leaves
    // Number of neighbors: num_result
    MatrixXd nn_result(cloud_1.rows(), 3);
    kd_tree tree_1(3, cref(cloud_1), max_leaf);
    tree_1.index -> buildIndex();

    for (int i = 0; i < cloud_1.rows(); i++)
    {
        vector<double> point{cloud_2(i, 0), cloud_2(i, 1), cloud_2(i, 2)};
        unsigned long nn_index = 0; // Index of the closes neighbor will be stored here
        double error = 0; // Error term between the two points
        KNNResultSet<double> result(num_result); // Result of NN search for a single point
        result.init(&nn_index, &error);

        tree_1.index -> findNeighbors(result, &point[0], SearchParams(10)); // Run KNN search

        // Assign to container matrix
        nn_result(i, 0) = (double)i;
        nn_result(i, 1) = (double)nn_index;
        nn_result(i, 2) = error;
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
    // Short implementation of rigid body motion on all the point of an n*3 point cloud
    cloud *= R;
    cloud.rowwise() += t.transpose();
}

double calc_error(const MatrixXd& cloud_1, const MatrixXd& cloud_2, bool mean)
{
    if(mean) // Return the mean squared error between two point clouds
    {
        double mse = (cloud_1 - cloud_2).array().pow(2).sum() / (double)cloud_1.rows();
        cout << "mse=" << mse << endl;
        return mse;
    }
    else // Return the sum of squared residuals between two point clouds
    {
        double error = (cloud_1 - cloud_2).array().pow(2).sum();
        cout << error << endl;
        return error;
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
        T.block<3,3>(0,0) *= R;
        T.block<3,1>(0,3) += t;

        transform_cloud(cloud_1, R, t);

        // Compute the mean squared error
        double error = calc_error(cloud_1, cloud_2, true);

        // Check for convergence
        if (error < icp_error_t)
        {
            cout << "         ------[ ICP Converged! ]------" << endl;
            cout << "Error = " << calc_error(cloud_1, cloud_2, true) << endl;
            output_clouds(cloud_1, cloud_2, "ICP"); // Write the clouds into a file
            return T;
        }
    }

    cout << " ------[ ICP Reached max. iterations! ]------" << endl;
    cout << "Mean Squared Error = " << calc_error(cloud_1, cloud_2, true) << endl;
    output_clouds(cloud_1, cloud_2, "ICP"); // Write the clouds into a file
    return T;
}

MatrixXd sort_matrix(MatrixXd mat)
{
    // Sorts a a matrix by one of the columns
    // Columns of an mat matrix are {point_1, point_2, error}
    // Note: the column index must be filled correctly or it will be very hard to debug
    // Move all the rows of the matrix into a vector
    vector<VectorXd> vec;
    for (int i = 0; i < mat.rows(); i++)
    {
        vec.emplace_back(mat.row(i));
    }
    // This is a lambda that defines the < operation between two vectors
    sort(vec.begin(), vec.end(), [] (VectorXd const& t1, VectorXd const& t2)
    {
        return t1(2) < t2(2); // The index in parentheses defines the key column
    });
    // Put the rows from the vector back into the matrix
    for (int i = 0; i < mat.rows(); i++)
    {
        mat.row(i) = vec[i];
    }
    return mat;
}

MatrixXd trim(const MatrixXd& mat, const double& overlap)
{
    // Trims an Eigen matrix to create a smaller matrix
    // A new matrix gets created and the data gets copied over
    int trimmed_len = (int)(overlap * (double)mat.rows());
    MatrixXd result(trimmed_len, mat.cols());
    for(int i = 0; i < trimmed_len; i++)
    {
        result.row(i) = mat.row(i);
    }
    return result;
}

void reorder_2(MatrixXd& cloud_1, MatrixXd& cloud_2, const MatrixXd& indices)
{
    // Reorder the rows of a matrix to match the ones given by the nearest neighbor search
    // This is used by the TR-ICP algorithm and reorders two clouds based on indices
    // Note: the indices matrix is always shorter in dimension 0 than the clouds
    MatrixXd result_1 = MatrixXd::Ones(indices.rows(), 3);
    MatrixXd result_2 = MatrixXd::Ones(indices.rows(), 3);
    for (int i = 0; i < indices.rows(); i++)
    {
        result_1.row(i) = cloud_1.row((int)indices(i, 0)); // Add to position i on cloud 1
        result_2.row(i) = cloud_2.row((int)indices(i, 1)); // Add to position i on cloud 2
    }
    cloud_1 = result_1;
    cloud_2 = result_2;
}

double obj_func(double x, MatrixXd nn)
{
    // Objective function for golden section search.
    // e(xi) / e^(1+lambda); lambda=2
    nn = trim(nn, x);
    return nn.col(2).mean() / pow(x, 1 + lambda);
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

Matrix4d tr_icp(MatrixXd cloud_1, MatrixXd cloud_2)
{
    // Trimmed iterative closest point algorithm implementation
    double error_prev = 1e18; // Set the previous error to a large number
    double error; // This variable will hold the error for reference
    const MatrixXd cloud_2_or = cloud_2; // The original model cloud
    MatrixXd cloud_1_tr = cloud_1; // The transformed source cloud
    Matrix4d T = Matrix4d::Identity(); // Transformation matrix

    for (int i = 0; i < tricp_iter; i++)
    {
        MatrixXd nn = nn_search(cloud_1, cloud_2); // Compute nearest neighbors

        nn = sort_matrix(nn); // Sort the NN search result by error

        xi = golden_section_search(0.1, 0.9, 0.001, nn); // G-search [min, max, tolerance, nn]

        nn = trim(nn, xi); // Trim by minimum overlap parameter

        reorder_2(cloud_1, cloud_2, nn); // Reorder based on indices defined in NN object

        error = calc_error(cloud_1, cloud_2, true); // Compute the squared error

        pair<Matrix3d, Vector3d> transform = estimate_transformation(cloud_1, cloud_2);
        Matrix3d R = transform.first; // Rotation matrix
        Vector3d t = transform.second; // Translation vector

        // Update the transformation matrix
        T.block<3,3>(0,0) *= R;
        T.block<3,1>(0,3) += t;

        transform_cloud(cloud_1_tr, R, t); // Transform the point cloud

        cloud_1 = cloud_1_tr; // Assign transformed cloud to trimmed cloud 1
        cloud_2 = cloud_2_or; // Assign original cloud to timmed cloud 2

        // Convergence test
        if (error < tricp_error_t || abs(error - error_prev) < tricp_error_change_t)
        {
            cout << "         ------[ TR-ICP Converged! ]------" << endl;
            cout << "Mean Squared Error = " << error << endl;
            cout << "Change of Error = " << abs(error - error_prev) << endl;
            output_clouds(cloud_1, cloud_2, "TR-ICP"); // Write the clouds into a file
            return T;
        }
        error_prev = error; // Update error change term
    }

    cout << "   ----[ TR-ICP Reached max. iterations! ]----" << endl;
    cout << "Mean Squared Error = " << error << endl;
    cout << "Change of Error = " << abs(error - error_prev) << endl;
    output_clouds(cloud_1, cloud_2, "TR-ICP"); // Write the clouds into a file
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

