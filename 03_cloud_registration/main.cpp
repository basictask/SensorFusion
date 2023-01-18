//=================[ Made by Daniel Kuknyo ]=================//
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "main.h"
#include "nanoflann.hpp"
#include </usr/include/eigen3/Eigen/Dense>
#include </usr/local/include/opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace happly;
using namespace nanoflann;

//----------[ Parameters set by program ]----------
unsigned long n_rows; // Number of rows to process
string cloud_name; // Name of cloud for outputting results
double xi; // Maximum overlap parameter for tr-icp. N_po = xi * N_p
float timenow; // Timing variable

int main(int argc, char** argv)
{
    // Define the variables for point clouds
    vector<Point3d> vector_1;
    vector<Point3d> vector_2;
    MatrixXd cloud_1;
    MatrixXd cloud_2;

    // Read point clouds
    if(argc==3 && !apply_init_transformation) // NOLINT
    {
        // If 2 args are given read both clouds from file
        vector_1 = read_pointcloud(argv[1]);
        vector_2 = read_pointcloud(argv[2]);
        n_rows = min(vector_1.size(), vector_2.size());
        cloud_1 = vector2mat(vector_1); // Source cloud
        cloud_2 = vector2mat(vector_2); // Target cloud
    }
    else
    {
        // In case 1 arg is given transform the original cloud to become the model cloud
        vector_2 = read_pointcloud(argv[2]);
        n_rows = vector_2.size();
        cloud_2 = vector2mat(vector_2); // Target cloud
        cloud_1 = apply_init_transform(cloud_2); // Create source with applied transformation
    }

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
    Matrix4d T_icp = icp(cloud_1, cloud_2);
    cout << "Transformation matrix between point clouds: " << endl << T_icp << endl;

    // Run the TR-ICP algorithm
    cout << "===================[ TR-ICP ]===================" << endl;
    Matrix4d T_tr_icp = tr_icp(cloud_1, cloud_2);
    cout << "Transformation matrix between point clouds: " << endl << T_tr_icp << endl;

    cout << "====================[ Done ]====================" << endl;
    
    return 0;
}

MatrixXd apply_init_transform(MatrixXd cloud)
{
    // Applies rotation, translation and Gaussian noise to the target point cloud in order to create the source cloud
    // Modify global variaables lvl_noise, lvl_translation and lvl_rotation defined in main.h to change how it behaves
    // Create the rotation matrix
    if(lvl_noise == 0 && lvl_translation == 0 && lvl_rotation == 0) // NOLINT
    {
        cout << "Unable to perform initial transformation. All the params are set to 0 [main.h]." << endl;
        exit(EXIT_FAILURE);
    }
    // Define the rotation matrix. Note: This is a rotation around the y axis.
    Matrix3d R;
    if(lvl_rotation > 0.)
    {
        R << cos(lvl_rotation), 0, sin(lvl_rotation),
                             0, 1, 0,
            -sin(lvl_rotation), 0, cos(lvl_rotation);
    }
    else
    {
        R = Matrix3d::Identity();
    }
    // Create the translation vector. Note: This is a translation on the y axis
    Vector3d t;
    if(lvl_translation > 0.)
    {
        t << 0, lvl_translation, 0;
    }
    else
    {
        t.setZero();
    }
    // Apply transformation
    transform_cloud(cloud, R, t);
    // Add noise if needed
    if(lvl_noise > 0.)
    {
        default_random_engine generator; // NOLINT
        normal_distribution<double> dist(0, lvl_noise); // Create Gaussian object
        for(int i = 0; i < cloud.rows(); i++)
        {
            for(int j = 0; j < cloud.cols(); j++)
            {
                cloud(i, j) += dist(generator); // Add Gaussian noise to the point
            }
        }
    }
    return cloud;
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
    double error; // This variable will hold the error for reference
    int iter = 0; // Iteration counter
    Matrix4d T = Matrix4d::Identity(); // Predicted transformation matrix
    Matrix4d T_true = estimate_T_true(cloud_1, cloud_2); // True transformation matrix (for logging)
    time_t start, end; // Time variables
    time(&start); // Time starts here

    for(int i = 0; i < icp_iter; i++)
    {
        MatrixXd nn = nn_search(cloud_1, cloud_2); // Compute nearest neighbors

        cloud_1 = reorder(cloud_1, nn); // Reorder points

        // Estimate the transformation between the point clouds
        pair<Matrix3d, Vector3d> transform = estimate_transformation(cloud_1, cloud_2);
        Matrix3d R = transform.first; // Rotation matrix
        Vector3d t = transform.second; // Translation vector

        // Update the transformation matrix
        T.block<3,3>(0,0) *= R;
        T.block<3,1>(0,3) += t;

        transform_cloud(cloud_1, R, t); // Apply the transformation to the point cloud
        error = calc_error(cloud_1, cloud_2, true); // Compute the mean squared error
        iter++; // Increase step counter

        // Check for convergence
        if (error < icp_error_t)
        {
            cout << "        ------[ ICP Converged! ]------" << endl;
            break;
        }
        if(i == icp_iter - 1)
        {
            cout << " ------[ ICP Reached max. iterations! ]------" << endl;
        }
    }

    time(&end); // End timer
    timenow = float(end - start); // Calculate duration of the algorithm
    cout << "Mean Squared Error = " << error << endl;
    output_clouds(cloud_1, cloud_2, "ICP"); // Write the clouds into a file
    log_execution(cloud_1, cloud_2, "ICP", error, iter, (iter < icp_iter - 1), T_true, T); // Write into log file
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

void reorder_trim(MatrixXd& cloud_1, MatrixXd& cloud_2, const MatrixXd& indices, const double& overlap)
{
    // Reorder the rows of a matrix to match the ones given by the nearest neighbor search
    // This is used by the TR-ICP algorithm and reorders two clouds based on indices
    // Note: the indices matrix is always shorter in dimension 0 than the clouds

    int trimmed_len = (int)(overlap * (double)indices.rows()); // Calculate trimmed number of points
    MatrixXd result_1(trimmed_len, 3); // Create container for trimmed cloud 1
    MatrixXd result_2(trimmed_len, 3); // Create container for trimmed cloud 2

    for (int i = 0; i < trimmed_len; i++)
    {
        result_1.row(i) = cloud_1.row((int)indices(i, 0)); // Add to position i on cloud 1
        result_2.row(i) = cloud_2.row((int)indices(i, 1)); // Add to position i on cloud 2
    }
    cloud_1 = result_1;
    cloud_2 = result_2;
}

Matrix4d tr_icp(MatrixXd cloud_1, MatrixXd cloud_2)
{
    // Trimmed iterative closest point algorithm implementation
    double error; // This variable will hold the error for reference
    double error_prev = 1e18; // Set the previous error to a large number
    int iter = 0; // Iteration counter
    Matrix4d T = Matrix4d::Identity(); // Predicted transformation matrix
    Matrix4d T_true = estimate_T_true(cloud_1, cloud_2); // True transformation matrix (for logging)
    MatrixXd cloud_1_tr = cloud_1; // The transformed source cloud
    const MatrixXd cloud_2_or = cloud_2; // The original model cloud
    time_t start, end; // Time variables
    time(&start); // Time starts here

    for(int i = 0; i < tricp_iter; i++)
    {
        MatrixXd nn = nn_search(cloud_1, cloud_2); // Compute nearest neighbors

        nn = sort_matrix(nn); // Sort the NN search result by error

        xi = golden_section_search(0.1, 0.9, 0.001, nn); // G-search [min, max, tolerance, nn]

        reorder_trim(cloud_1, cloud_2, nn, xi); // Reorder and trim based on indices defined in NN object

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
        iter++; // Increase step counter

        // Convergence test
        if(error < tricp_error_t || abs(error - error_prev) < tricp_error_change_t)
        {
            cout << "        ------[ TR-ICP Converged! ]------" << endl;
            break;
        }
        if(i == tricp_iter - 1)
        {
            cout << "   ----[ TR-ICP Reached max. iterations! ]----" << endl;
        }
        error_prev = error; // Update error change term
    }

    time(&end); // End timer
    timenow = float(end - start); // Calculate duration of the algorithm
    cout << "Mean Squared Error = " << error << endl;
    cout << "Change of Error = " << abs(error - error_prev) << endl;
    output_clouds(cloud_1, cloud_2, "TR-ICP"); // Write the clouds into a file
    log_execution(cloud_1, cloud_2, "TR-ICP", error, iter, (iter < tricp_iter - 1), T_true, T); // Write into log file
    return T;
}

