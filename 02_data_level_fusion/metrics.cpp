#include <opencv2/opencv.hpp>
#include <iostream>

# define GAUSSIAN 0

using namespace std;
using namespace cv;

// SSD over MSD
float CompareGtSSD(Mat image, Mat GT)
{
    image.convertTo(image, CV_32F, 1.0 / 255, 0);
    GT.convertTo(GT, CV_32F, 1.0 / 255, 0);
    
    float ssd = 0;
    float count = image.size().height * image.size().width;
    for(int i = 0; i < image.size().height; i++)
    {
        for(int j = 0; j < image.size().width; j++)
        {
            ssd += (image.at<uchar>(i,j) - GT.at<uchar>(i,j)) * (image.at<uchar>(i,j) - GT.at<uchar>(i,j));
        }
    }

    float mse = ssd / count;
    return mse;
}

// Normalized cross correlation to compare images
float CompareGTNCC(Mat image, Mat GT)
{
    float count = image.size().height * image.size().width;
    float i_avg, g_avg, i_sig, g_sig, ncc = 0;

    for(int i = 0; i < image.size().height; i++)
    {
        for(int j = 0; j < image.size().width; j++)
        {
            i_avg += image.at<uchar>(i, j);
            g_avg += GT.at<uchar>(i, j);
        }
    }

    i_avg /= count;
    g_avg /= count;

    for(int i = 0; i < image.size().height; i++)
    {
        for(int j = 0; j < image.size().width; j++)
        {
            i_sig += pow(i_avg - image.at<uchar>(i, j), 2);
            g_sig += pow(g_avg - GT.at<uchar>(i, j), 2);
        }
    }

    i_sig = sqrt(i_sig / (count - 1));
    g_sig = sqrt(g_sig / (count - 1));

    for(int i = 0; i < image.size().height; i++)
    {
        for(int j = 0; j < image.size().width; j++)
        {
            ncc += ((image.at<uchar>(i,j) - i_avg) * (GT.at<uchar>(i,j) - g_avg));
        }
    }

    ncc /= sqrt(i_sig * g_sig);
    ncc /= count;

    return ncc;
}

// Structural similarity index to compare two images
float CompareGtSSIM(cv::Mat image1, cv::Mat image2, int kernel_size) 
{
    static const double C1 = 6.5025;
    static const double C2 = 58.5225;

    cv::Mat img1_f, img2_f;
    image1.convertTo(img1_f, CV_32F);
    image2.convertTo(img2_f, CV_32F);
    cv::Mat tmp;

    /* Perform mean filtering on image using boxfilter */
    cv::Mat img1_avg, img2_avg;
    cv::boxFilter(img1_f, img1_avg, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
    cv::boxFilter(img2_f, img2_avg, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
    GaussianBlur(img1_f, img1_avg, cv::Size(kernel_size, kernel_size), 1.5);
    GaussianBlur(img2_f, img2_avg, cv::Size(kernel_size, kernel_size), 1.5);
#endif // GAUSSIAN
    cv::Mat img1_avg_sqr = img1_avg.mul(img1_avg);
    cv::Mat img2_avg_sqr = img2_avg.mul(img2_avg);

    /* Calculate variance map */
    cv::Mat img1_1 = img1_f.mul(img1_f);
    cv::Mat img2_2 = img2_f.mul(img2_f);
    cv::boxFilter(img1_1, tmp, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
    GaussianBlur(img1_1, tmp, cv::Size(kernel_size, kernel_size), 1.5);
#endif
    cv::Mat img1_var = tmp - img1_avg_sqr;
    cv::boxFilter(img2_2, tmp, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
    GaussianBlur(img2_2, tmp, cv::Size(kernel_size, kernel_size), 1.5);
#endif
    cv::Mat img2_var = tmp - img2_avg_sqr;

    /* Calculate covariance map */
    cv::Mat src_mul = img1_f.mul(img2_f);
    cv::Mat avg_mul = img1_avg.mul(img2_avg);
    cv::boxFilter(src_mul, tmp, -1, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1), true, cv::BORDER_DEFAULT);
#if GAUSSIAN
    GaussianBlur(src_mul, tmp, cv::Size(kernel_size, kernel_size), 1.5);
#endif
    cv::Mat covariance = tmp - avg_mul;

    auto num = ((2 * avg_mul + C1).mul(2 * covariance + C2));
    auto den = ((img1_avg_sqr + img2_avg_sqr + C1).mul(img1_var + img2_var + C2));

    cv::Mat ssim_map;
    cv::divide(num, den, ssim_map);

    cv::Scalar mean_val = cv::mean(ssim_map);
    float mssim = (float)mean_val.val[0];

    return mssim;
}
