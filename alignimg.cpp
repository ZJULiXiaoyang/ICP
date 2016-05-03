
#include"icp.h"
#include<opencv2/opencv.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<features.h>

void alignimg(cv::Mat& rgb_image1, cv::Mat& rgb_image2, cv::Mat& depth_image1, cv::Mat& depth_image2, cv::Mat& R10, cv::Mat& t10,
              string keypointFile1,string keypointFile2,string descriptorFile1,string descriptorFile2,int flag)
{
    cv::initModule_nonfree();//just to state the right of the function which may not be free
    vector<cv::KeyPoint> keypoint1, keypoint2;
    vector<cv::Point3f> keypoint3d1, keypoint3d2;//there are still Point3i、Point3d
    vector<cv::Point2f> kp1, kp2;
    vector<cv::DMatch> matches;
    cv::Mat descriptor1, descriptor2;

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(MATCHER_NAME);

    if (flag)
    {
        loadfeaturepoint(keypointFile1.c_str(),keypoint1);
        loadfeaturepoint(keypointFile2.c_str(),keypoint2);//load the keypoint
        loadDescriptor(descriptorFile1.c_str(),descriptor1,192);
        loadDescriptor(descriptorFile2.c_str(),descriptor2,192);//load the descriptor

        findDepth(keypoint1,keypoint3d1,depth_image1,kp1);
        findDepth(keypoint2,keypoint3d2,depth_image2,kp2);//LIOP
    }else
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(DETECTOR_NAME);
        cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create(DESCRIPTOR_NAME);
        detector->detect(rgb_image1, keypoint1);
        detector->detect(rgb_image2, keypoint2);

        findDepth(keypoint1,keypoint3d1,depth_image1,kp1);
        findDepth(keypoint2,keypoint3d2,depth_image2,kp2);

        descriptor->compute(rgb_image1, keypoint1, descriptor1);
        descriptor->compute(rgb_image2, keypoint2, descriptor2);//SURF
    }
    matcher->match( descriptor1, descriptor2, matches);//feature matching

    vector<cv::KeyPoint> matchedKeyPoint1, matchedKeyPoint2;
    vector<cv::Point3f> matchedKP3d1, matchedKP3d2;
    vector<cv::Point2f> matchedKP1, matchedKP2;
    for (vector<cv::DMatch>::iterator iter = matches.begin(); iter != matches.end(); ++iter)
    {
        matchedKeyPoint1.push_back(keypoint1[iter->queryIdx]);
        matchedKeyPoint2.push_back(keypoint2[iter->trainIdx]);
        matchedKP3d1.push_back(keypoint3d1[iter->queryIdx]);
        matchedKP3d2.push_back(keypoint3d2[iter->trainIdx]);
        matchedKP1.push_back(kp1[iter->queryIdx]);
        matchedKP2.push_back(kp2[iter->trainIdx]);
    }

    vector<uchar> inliersMask;
    cv::Mat homography = findHomography(matchedKP1, matchedKP2, CV_RANSAC,3, inliersMask);

    vector<cv::KeyPoint> keypointInlier1, keypointInlier2;
    vector<cv::Point2f> leftInlier,rightInlier;
    vector<cv::Point3f> left3dInlier,right3dInlier;
    vector<cv::DMatch> inlierMatch;
    int index = 0;
    for (unsigned i = 0; i < matches.size(); i++) {
        if (inliersMask[i] != 0){
            keypointInlier1.push_back(matchedKeyPoint1[i]);
            keypointInlier2.push_back(matchedKeyPoint2[i]);
            leftInlier.push_back(matchedKP1[i]);
            rightInlier.push_back(matchedKP2[i]);
            left3dInlier.push_back(matchedKP3d1[i]);
            right3dInlier.push_back(matchedKP3d2[i]);
            matches[i].trainIdx = index;
            matches[i].queryIdx = index;
            inlierMatch.push_back(matches[i]);
            index++;
        }
    }
    matches = inlierMatch;
    cv::Mat matchimg;
    drawMatches(rgb_image1, keypointInlier1, rgb_image2, keypointInlier2, matches, matchimg );
    imshow("Match", matchimg);
    cv::waitKey();

    cout << "computing pose ..." <<endl;
    float K[3][3];
    K[0][0] = FOCAL; K[0][1] = 0.0f; K[0][2] = CX;
    K[1][0] = 0.0f; K[1][1] = FOCAL; K[1][2] = CY;
    K[2][0] = 0.0f; K[2][1] = 0.0f; K[2][2] = 1.0f;
    cv::Mat cameraMatrix(3, 3, CV_32FC1, K);
    cv::Mat R10vec;
    cv::Mat inliers;
    solvePnPRansac(right3dInlier, leftInlier, cameraMatrix, cv::noArray(), R10vec, t10,false,300,10.0f,99999999,inliers);
    Rodrigues(R10vec, R10);
    vector<cv::DMatch> inliermatches;
    for (int i=0; i<inliers.rows; ++i)
        inliermatches.push_back(matches[inliers.at<int>(i)]);
    cv::Mat finallMatch;
    drawMatches(rgb_image1, keypointInlier1, rgb_image2, keypointInlier2, inliermatches, finallMatch );
    imshow("FinalMatch",finallMatch);
   // cv::waitKey();

}

