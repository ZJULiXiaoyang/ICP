#include "icp.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/flann.hpp>
#include <highgui.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/flann.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include<sstream>

#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace cv;

string int2str(int num)
{
    stringstream ss;
    ss<<num;
    return ss.str();
}

int main(int argc,char* argv[])
{
    string num1,num2;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    vector<Eigen::Matrix4f> TRMatrix1;
    vector<Eigen::Matrix4f> TRMatrix2;
    vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_final (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_original(new pcl::PointCloud<pcl::PointXYZRGB>());
    int startNum=atoi(argv[3]);
    int dset=atoi(argv[4]);
    int endNum=atoi(argv[5]);
    for(int i=startNum;i<endNum;i=i+dset)
    {
        num1=int2str(i);
        num2=int2str(i+dset);

        string filePath=argv[2];
        string pathImg1=filePath+"/rgb_"+num1+".png";
        string pathImg2=filePath+"/rgb_"+num2+".png";
        string pathDepth1=filePath+"/depth_"+num1+".png";
        string pathDepth2=filePath+"/depth_"+num2+".png";
        string pathkeyPoint1=filePath+"/keypoint"+num1+".txt";
        string pathkeyPoint2=filePath+"/keypoint"+num2+".txt";
        string pathdescriptor1=filePath+"/descriptor"+num1+".txt";
        string pathdescriptor2=filePath+"/descriptor"+num2+".txt";

        cv::Mat depth_image1=imread(pathDepth1,CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat depth_image2=imread(pathDepth2,CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat rgb_image1=imread(pathImg1);
        cv::Mat rgb_image2=imread(pathImg2);//read the image

        string keypointFile1=pathkeyPoint1;
        string descriptoFile1=pathdescriptor1;
        string keypointFile2=pathkeyPoint2;
        string descriptoFile2=pathdescriptor2;//the keypoint and descriptor file path

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1=depth2cloud(rgb_image1,depth_image1);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2=depth2cloud(rgb_image2,depth_image2);//get the pointcloud
        /*viewer = rgbVis(cloud1,cloud2,1);
        viewer->resetCamera ();
        viewer->spin ();
        while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }*/


        if(i==startNum)
            *cloud_final=*cloud1;
        clouds.push_back(cloud2);

        Eigen::Matrix4f transformationINSERT;//define the transformation matrix

        cv::Mat R10,t10;
        cout<<"R:"<<endl<<R10<<endl;
        cout<<"T:"<<endl<<t10<<endl;
        alignimg(rgb_image1,rgb_image2,depth_image1,depth_image2,R10,t10,keypointFile1,keypointFile2,descriptoFile1,descriptoFile2,atoi(argv[1]));//get the initial transformation matrix
        float M[4][4];
        M[0][0] = R10.at<double>(0,0); M[0][1] = R10.at<double>(0,1); M[0][2] = R10.at<double>(0,2);M[0][3]=t10.at<double>(0,0);
        M[1][0] = R10.at<double>(1,0); M[1][1] = R10.at<double>(1,1); M[1][2] = R10.at<double>(1,2);M[1][3]=t10.at<double>(1,0);
        M[2][0] = R10.at<double>(2,0); M[2][1] = R10.at<double>(2,1); M[2][2] = R10.at<double>(2,2);M[2][3]=t10.at<double>(2,0);
        M[3][0]=0.0f;M[3][1]=0.0f;M[3][2]=0.0f;	M[3][3]=1.0f;

        cv::Mat transformMatrix(4,4,CV_32FC1,M);
        cv2eigen(transformMatrix,transformationINSERT);
        cout<<transformationINSERT<<endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrot(new pcl::PointCloud<pcl::PointXYZRGB>());
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrot1(new pcl::PointCloud<pcl::PointXYZRGB>());
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrot2(new pcl::PointCloud<pcl::PointXYZRGB>());
        //pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::transformPointCloud(*cloud2,*cloudrot,transformationINSERT);
        TRMatrix1.push_back(transformationINSERT);
        /**sourceCloud=*cloud1+*cloud2;
        *cloudrot1=*cloud1+*cloudrot;
        viewer = rgbVis(sourceCloud,cloudrot1);
        viewer->resetCamera ();
        viewer->spin ();*/


        //pcl::IterativeClosestPointNonLinear<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setTransformationEpsilon(1e-8);
        icp.setMaxCorrespondenceDistance(0.05f);
        //icp.setRANSACOutlierRejectionThreshold(0.05);
        icp.setMaximumIterations(30);
        icp.setEuclideanFitnessEpsilon (1);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudDownsample(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrotDownsample(new pcl::PointCloud<pcl::PointXYZRGB>());

        pcl::VoxelGrid<pcl::PointXYZRGB> vg;
        vg.setInputCloud(cloud1);
        vg.setLeafSize(0.1f, 0.1f, 0.1f);
        vg.filter(*cloudDownsample);

        vg.setInputCloud(cloudrot);
        vg.setLeafSize(0.1f, 0.1f, 0.1f);
        vg.filter(*cloudrotDownsample);


        icp.setInputTarget(cloudDownsample);
        icp.setInputSource(cloudrotDownsample);
        icp.align(*cloudrotDownsample);//save the resultant cloud after applying the icp.
        std::cout << "has converged:" << icp.hasConverged() << " score: " <<icp.getFitnessScore(0.5) << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl;//output the transformation matrix
        /*Eigen::Matrix4f Ti=Eigen::Matrix4f::Identity();
        Ti=icp.getFinalTransformation()*Ti;
        cout<<"Ti:"<<endl<<Ti<<endl;*/

        Eigen::Matrix4f tmp;
        tmp=icp.getFinalTransformation()*TRMatrix1[(i-startNum)/dset];
        TRMatrix2.push_back(tmp);



        // pcl::transformPointCloud(*cloud2,*cloudrot1,TRMatrix1[i-startNum]);
        /*cloudrot2=*cloudrot1+*cloud1;
        viewer = rgbVis(sourceCloud,cloudrot2);
        viewer->resetCamera ();
        viewer->spin ();
        pcl::transformPointCloud(*cloudrot1,*cloudrot1,icp.getFinalTransformation());
        *cloudrot1+=*cloud1;
        viewer = rgbVis(sourceCloud,cloudrot1);
        viewer->resetCamera ();
        viewer->spin ();*/
    }

    for(vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>::iterator it=clouds.begin();it!=clouds.end();it++)
    {
        *cloud_original+=*(*it);
    }
    *cloud_original+=*cloud_final;//get the original pointcloud

    int index=0;
    for(vector<Eigen::Matrix4f>::iterator it=TRMatrix2.begin();it!=TRMatrix2.end();it++)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp (new pcl::PointCloud<pcl::PointXYZRGB>);
        Eigen::Matrix4f M=Eigen::Matrix4f::Identity();
        int i=0;
        for(vector<Eigen::Matrix4f>::iterator jt=TRMatrix2.begin();jt<=it;jt++)
        {
            M*=(*jt);
        }

        pcl::transformPointCloud(*(clouds[index]),*tmp,M);
        index++;
        *cloud_final+=*tmp;
    }

    viewer = rgbVis(cloud_original,cloud_final,1);
    viewer->resetCamera ();
    viewer->spin ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

}
