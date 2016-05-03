#define DETECTOR_NAME "SURF"
#define DESCRIPTOR_NAME "SURF"
#define MATCHER_NAME "BruteForce"
#define FOCAL 575.0f
#define CX 319.5f
#define CY 239.5f
#define HEIGHT 480
#define WIDTH 640
#define depth_style 1000.0f
#include<string>
#include<vector>
using namespace std;

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include<opencv2/opencv.hpp>

enum{
    STYLE_SURF=0,
    STYLE_LIOP=1
};

pcl::PointCloud<pcl::PointXYZRGB>::Ptr depth2cloud( cv::Mat rgb_image, cv::Mat depth_image ) ;//get the pointcloud from depth image and rgb image

void loadfeaturepoint(const char* featurefile,std::vector< cv::KeyPoint > & keypoints);//load keypoints

void alignimg(cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,cv::Mat&,string,string,string,string,int flag);//get the transformation matrix

void loadDescriptor(const char* descriptorfile,cv::Mat &descriptor,int descriptorSize);//load the descriptors

void findDepth( vector<cv::KeyPoint>& keypoint, vector<cv::Point3f>& keypoint3d, cv::Mat& depth_image, vector<cv::Point2f>& kp);//get the pointcloud of keypoint

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ,pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr ,int );

