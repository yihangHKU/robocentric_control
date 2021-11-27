#include <ros/ros.h>
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "vision_detect.hpp"
using namespace cv; 
using namespace std;

// hello realsense 

int maxCorners = 20;
int maxTrackbar = 100;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
Mat color_, gray_;
bool blob_init = false;
Ptr<SimpleBlobDetector> detector;
bool detect_init = false;
bool target_update = false;
bool obstacle_update = false;
int blob_target_num = 1;
// #define blob_detect_mode two_target
target_point_t targets[2];
target_point_t obstacle;

sensor_msgs::Image color_frame;
sensor_msgs::Image depth_frame;
sensor_msgs::CameraInfo color_info;
geometry_msgs::PoseArray gap_array;
geometry_msgs::PoseStamped gap_pose;
geometry_msgs::PoseStamped obs_pose;

rs2::frameset frames;
rs2::align align_to_color(RS2_STREAM_COLOR);

void align_fun()
{   
    // cout << "align fun begin" << endl;
    frames = align_to_color.process(frames);
    // cout << "align fun end" << endl;
}

void blob_detect_fun()
{   
    target_update = false;
    obstacle_update = false;
    vector<KeyPoint> keypoints; 
    cvtColor( color_, gray_, COLOR_BGR2GRAY );
    blur( gray_, gray_, Size(3,3) );
    detector->detect( gray_, keypoints);
        // namedWindow(blob_window);
    // std::cout << keypoints.size() << std::endl;
        // detector.detect( gray, keypoints);
    drawKeypoints( color_, keypoints, color_, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        // Canny(gray, imgcanny_blur, 100, 200, 3, true);
    

    switch (blob_target_num)
    {
    case 1: // one target point detect
        if (!detect_init)
        {
            if (keypoints.size() == 1)
            {
                targets[0].kp = keypoints.at(0);
                keypoints.clear();
                detect_init = true; 
                target_update = true;
                int i = 0;
                char num[1];
                sprintf(num, "%d", i);
                cv::putText(color_,string(num),targets[i].kp.pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
                cout << "detect init" << endl; 
            }
        }
        else if (keypoints.size() >= 1)
        {
            float min_size = targets[0].kp.size;
            bool kp_match = true;
            kp_match = nearest_pixel_find(targets[0].kp, keypoints, min_size);
            if (kp_match)
            {   
                int i = 0;
                char num[1];
                sprintf(num, "%d", i);
                cv::putText(color_,string(num),targets[i].kp.pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
                target_update = true;
            }
            else
            {
                detect_init = false;
                cout << "detect false" << endl;
            }
        }
        break;
    
    case 2: // two target points and one may exist obstacle detect
        if (!detect_init)
        {
            if (keypoints.size() == 2 && sqrt(pow(keypoints.at(0).pt.x - keypoints.at(1).pt.x, 2) + pow(keypoints.at(0).pt.y - keypoints.at(1).pt.y, 2)) < 150.0f)
            {
                if (keypoints.at(0).pt.x < keypoints.at(1).pt.x)
                {
                    targets[0].kp = keypoints.at(0);
                    targets[1].kp = keypoints.at(1);
                }
                else
                {
                    targets[0].kp = keypoints.at(1);
                    targets[1].kp = keypoints.at(0);
                }  
                keypoints.clear();
                detect_init = true; 
                target_update = true;
                for (int i = 0; i < 2; i++)
                {
                    char num[1];
                    sprintf(num, "%d", i);
                    cv::putText(color_,string(num),targets[i].kp.pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
                }
                cout << "detect init" << endl;  
            }
            else
            {
                // cout << "detect init error" << endl;
            }
        }
        else if (keypoints.size() >= 2)
        {   
            float min_size = min(targets[0].kp.size, targets[1].kp.size);
            bool kp_match = true;
            for (int i = 0; i < 2; i++)
            {
                kp_match = kp_match && nearest_pixel_find(targets[i].kp, keypoints, min_size);
            }
            if (kp_match)
            {
                if(targets[0].kp.pt.x > targets[1].kp.pt.x)
                {   
                    KeyPoint middle = targets[0].kp;
                    targets[0].kp = targets[1].kp;
                    targets[1].kp = middle;
                }
                for (int i = 0; i < 2; i++)
                {
                    char num[1];
                    sprintf(num, "%d", i);
                    cv::putText(color_,string(num),targets[i].kp.pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
                }
                target_update = true;
            }
            else
            {
                detect_init = false;
                cout << "detect false" << endl;
            }
        }
        // else
        // {
        //     // detect_init = false;
        // }

        if(!keypoints.empty())
        {
            float obs_size = keypoints.at(0).size;
            int obs_index = 0;
            for (int i = 0; i < keypoints.size(); i++)
            {
                if(keypoints.at(i).size > obs_size)
                {
                    obs_size = keypoints.at(i).size;
                    obs_index = i;
                }
            }
            char num[1];
            sprintf(num, "%d", 2);
            cv::putText(color_,string(num), keypoints.at(obs_index).pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
            obstacle.kp = keypoints.at(obs_index);
            obstacle_update = true;
        }  
        break;
    default:
        break;
    }
}

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "vision_detect");
    ros::NodeHandle nh;

    ros::Publisher gap_pose_pub = nh.advertise<geometry_msgs::PoseArray>
            ("/robocentric/camera/gap_pose2", 100);
    ros::Publisher obs_pose_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("/robocentric/camera/ball_pos", 100);
    ros::Publisher color_pub = nh.advertise<sensor_msgs::Image>
            ("/robocentric/camera/color_raw", 100);
    sensor_msgs::Image img_msg;
    std_msgs::Header header;
    cv_bridge::CvImage img_bridge;
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::colorizer color_map;
    blob_detect_init(detector);
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
    cfg.enable_stream(RS2_STREAM_INFRARED, 848, 480, RS2_FORMAT_Y8, 90);
    pipe.start(cfg);
    // rs2::pipeline_profile profile = pipe.start();
    
    // rs2::align align_to_depth(RS2_STREAM_DEPTH);
    ros::Rate rate(50.0);
    fout_vision.open("/home/yihang/catkin_ws/debug/mat_vision.txt", std::ios::out);
    fout_depth.open("/home/yihang/catkin_ws/debug/mat_depth.txt", std::ios::out);
    Eigen::Matrix<float, 3, 1> last_pub_P{0.0, 0.0, 0.0};
    Eigen::Matrix<float, 3, 1> last_pub_P2{0.0, 0.0, 0.0};
    for(int i = 0; i < 10; i++)
    {
        frames = pipe.wait_for_frames();
        // ros::spinOnce();
    }
    // float depth_scale = get_depth_scale(profile.get_device());
    float depth_clipping_distance = 5.0;
    // std::cout << "depth_scale: " << depth_scale << std::endl;
    int i = 0;
    double total_time = 0.0;
    double total_time1 = 0.0;
    float last_target0_depth = 0.0;
    float last_target1_depth = 0.0;
    while(ros::ok()){ 
        frames = pipe.wait_for_frames();
        rs2::video_frame color_frame = frames.get_color_frame();
        int width = color_frame.get_width();
        int height = color_frame.get_height();
        Mat color(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        color_ = color;
        std::thread t1(blob_detect_fun);
        // double begin = ros::Time::now().toSec();
        align_fun();
        // std::thread t1(align_fun);
        // double end = ros::Time::now().toSec();
        // total_time1 += end - begin;
        // std::cout << "thread 1 id: " << t1.get_id() << std::endl;

        // blob detection
        i++;
        // double time1 = ros::Time::now().toSec();
        // std::cout << "thread 2 id: " << t2.get_id() << std::endl;
        // blob_detect(detector, gray, color_, depth_frame);
        // double time2 = ros::Time::now().toSec();
        // total_time += time2 - time1;

        // t1.join();

        rs2::depth_frame depth_frame = frames.get_depth_frame();
        // remove_background(color_frame, depth_frame, 0.001, depth_clipping_distance);
        // rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
        rs2_intrinsics depth_intrins = rs2::video_stream_profile(depth_frame.get_profile()).get_intrinsics();
        // rs2_intrinsics color_intrins = rs2::video_stream_profile(color_frame.get_profile()).get_intrinsics();
        cv::Matx33f cameraMatrix = {depth_intrins.fx, 0., depth_intrins.ppx, 0, depth_intrins.fy, depth_intrins.ppy, 0, 0 ,0};
        vector<float> distCoeffs = {depth_intrins.coeffs[0], depth_intrins.coeffs[1], depth_intrins.coeffs[2], depth_intrins.coeffs[3], depth_intrins.coeffs[4]};
        // cv::Matx33f cameraMatrix = {color_info.K[0], 0., color_info.K[2], 0., color_info.K[3], color_info.K[4], 0., 0., 0.};
        // vector<float> distCoeffs = color_info.D;
        // for (int i = 0; i<5; i++)
        // {    
        //     cout << "depth distcoeffs: " << distCoeffs[i] << endl;
        // }    
        // cout << "depth intrinsics ppx" << cameraMatrix << endl;
        // rs2::frame colorized_depth = color_map.colorize(depth_frame);
        Mat depth(Size(width, height), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        // Mat ir(Size(1280, 720), CV_8UC1, (void*)ir_frame.get_data(), Mat::AUTO_STEP);   
        // Mat imgcanny;
        bool depth_error = false;


        // gap detection 
        // depth_error = gap_detect(gray, depth_frame, color_);

        // houghcircles detection 
        // vector<Vec3f> circles;
        // HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 2, 200, 150, 80, 0, 50);
        // cout << "circle size: " << circles.size() << endl;
        // for (size_t i = 0; i < circles.size(); i++)
        // {
        //     Point2f center(circles[i][0],circles[i][1]);
        //     float radius = circles[i][2];
        //     circle(color_, center, 4, Scalar(0, 255, 0), CV_FILLED);
        //     circle(color_, center, radius, Scalar(255, 0, 0), 3, 8, 0);
        // }
        t1.join();
        switch (blob_target_num)
        {
        case 1:
            if (target_update)
            {
                get_depth(color_, depth_frame, targets[0]);
                if (targets[0].depth > 0.5 && (abs(targets[0].depth - last_target0_depth) < 0.5 || last_target0_depth < 0.1))
                {    
                    gap_pose.pose.position.x = targets[0].point[0];
                    gap_pose.pose.position.y = targets[0].point[1];
                    gap_pose.pose.position.z = targets[0].point[2];
                    gap_array.poses.push_back(gap_pose.pose);
                    gap_array.header.stamp = ros::Time::now();
                    gap_pose_pub.publish(gap_array); 
                    gap_array.poses.clear();
                    last_target0_depth = targets[0].depth;
                }
            }
            break;
        
        case 2: 
            if (target_update)
            {
                get_depth(color_, depth_frame, targets[0]);
                get_depth(color_, depth_frame, targets[1]);
                if (targets[0].depth > 0.5 && targets[1].depth > 0.5 && (abs(targets[0].depth - last_target0_depth) < 0.5 || last_target0_depth < 0.1) && (abs(targets[1].depth - last_target1_depth) < 0.5 || last_target1_depth < 0.1))
                {    
                    gap_pose.pose.position.x = targets[0].point[0];
                    gap_pose.pose.position.y = targets[0].point[1];
                    gap_pose.pose.position.z = targets[0].point[2];
                    gap_array.poses.push_back(gap_pose.pose);
                    gap_pose.pose.position.x = targets[1].point[0];
                    gap_pose.pose.position.y = targets[1].point[1];
                    gap_pose.pose.position.z = targets[1].point[2];
                    gap_array.poses.push_back(gap_pose.pose);
                    gap_array.header.stamp = ros::Time::now();
                    gap_pose_pub.publish(gap_array); 
                    gap_array.poses.clear();
                    last_target0_depth = targets[0].depth;
                    last_target1_depth = targets[1].depth;
                }
            }
            break;
        
        default:
            break;
        }
        

        if (obstacle_update)
        {
            get_depth(color_,depth_frame, obstacle);
            if (obstacle.depth > 0.1 && obstacle.depth < 4.0)
            {
                obs_pose.pose.position.x = obstacle.point[0];
                obs_pose.pose.position.y = obstacle.point[1];
                obs_pose.pose.position.z = obstacle.point[2];
                obs_pose.header.stamp = ros::Time::now();
                obs_pose_pub.publish(obs_pose);
            }
        }
        
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, color_);
        img_bridge.toImageMsg(img_msg);

        if (!depth_error)
        {
            color_pub.publish(img_msg);
        }
        
        rate.sleep();
    }
    // namedWindow("imgContour", CV_WINDOW_AUTOSIZE);
    // imshow("imgContour", imgcontour);
    // namedWindow( source_window );
    // imshow( source_window, color_);
    // waitKey(0);
    ros::spin();
    return 0;
}
