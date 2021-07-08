#include <ros/ros.h>
#include <iostream>
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

sensor_msgs::Image color_frame;
sensor_msgs::Image depth_frame;
sensor_msgs::CameraInfo color_info;
geometry_msgs::PoseArray gap_array;
geometry_msgs::PoseStamped gap_pose;


// void color_cb(const sensor_msgs::Image::ConstPtr &msg)
// {
//     cv_bridge::CvImagePtr cv_ptr;
//     try
//     {
//        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
//     }
//     catch (cv_bridge::Exception& e)
//     {
//         ROS_ERROR("cv_bridge exception: %s", e.what());
//         return;
//     }
// }
int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "vision_detect");
    ros::NodeHandle nh;

    ros::Publisher gap_pose_pub = nh.advertise<geometry_msgs::PoseArray>
            ("/robocentric/camera/gap_pose2", 100);
    ros::Publisher color_pub = nh.advertise<sensor_msgs::Image>
            ("/robocentric/camera/color_raw", 100);
    sensor_msgs::Image img_msg;
    std_msgs::Header header;
    cv_bridge::CvImage img_bridge;
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::colorizer color_map;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
    cfg.enable_stream(RS2_STREAM_INFRARED, 848, 480, RS2_FORMAT_Y8, 90);
    pipe.start(cfg);
    // rs2::pipeline_profile profile = pipe.start();
    rs2::frameset frames;
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    ros::Rate rate(100.0);
    // time_t t;
    // tm* local;
    // char buf[128] = {0};
    // t = time(NULL);
    // local = localtime(&t);
    // strftime(buf, 64, "%Y-%m-%d %H:%M:%S", local);
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
    rs2::align align_to_color(RS2_STREAM_COLOR);
    while(ros::ok()){ 
        frames = pipe.wait_for_frames();
        // double begin = ros::Time::now().toSec();
        frames = align_to_color.process(frames);
        // double end = ros::Time::now().toSec();
        // std::cout << "align time: " << end - begin << std::endl;
        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        // remove_background(color_frame, depth_frame, 0.001, depth_clipping_distance);
        // rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
        int width = depth_frame.get_width();
        int height = depth_frame.get_height();
        float dist_to_center = depth_frame.get_distance(width/2, height/2);
        // float dist_to_center = depth_frame.data[int(width/2)][int(height/2)];
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
        Mat color(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        color_ = color;
        Mat depth(Size(width, height), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        // Mat ir(Size(1280, 720), CV_8UC1, (void*)ir_frame.get_data(), Mat::AUTO_STEP);   
        Mat gray;
        // Mat imgcanny;
        cvtColor( color, gray, COLOR_BGR2GRAY );
        gray_ = gray;
        blur( gray, gray, Size(3,3) );
        bool depth_error = false;

        // blob detection
        if (!blob_init)
        {   
            blob_init = blob_detect_init(detector);
        }
        // blob_detect(detector, gray, color_, depth_frame);

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

        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, color_);
        img_bridge.toImageMsg(img_msg);
        if (!depth_error)
        {
            color_pub.publish(img_msg);
        }
        if (!P_buffer.empty() && !q_buffer.empty() && !P2_buffer.empty())
        {
            float distance = P_buffer[0].norm();
            int index = 0;
            for (int i=0; i<P_buffer.size(); i++)
            {
                if (P_buffer[i].norm() < distance)
                {
                    index = i;
                    distance = P_buffer[i].norm();
                }
            }
            std::cout << "gap index: " << index << std::endl;
            Eigen::Matrix<float, 3, 1> pub_P = P_buffer[index];
            Eigen::Matrix<float, 3, 1> pub_P2 = P2_buffer[index];
            Eigen::Quaternionf pub_q = q_buffer[index];
            std::cout << "P: " << pub_P << std::endl;
            // std::cout << "q: " << pub_q << std::endl;
            P_buffer.clear();
            P2_buffer.clear();
            q_buffer.clear();
            if (!depth_error)
            {
                gap_pose.pose.position.x = pub_P[0];
                gap_pose.pose.position.y = pub_P[1];
                gap_pose.pose.position.z = pub_P[2];
                gap_pose.pose.orientation.x = pub_q.x();
                gap_pose.pose.orientation.y = pub_q.y();
                gap_pose.pose.orientation.z = pub_q.z();
                gap_pose.pose.orientation.w = pub_q.w();
                gap_array.poses.push_back(gap_pose.pose);
                gap_pose.pose.position.x = pub_P2[0];
                gap_pose.pose.position.y = pub_P2[1];
                gap_pose.pose.position.z = pub_P2[2];
                gap_array.poses.push_back(gap_pose.pose);
                gap_array.header.stamp = ros::Time::now();
                gap_pose_pub.publish(gap_array); 
                gap_array.poses.clear();
            } 
            last_pub_P = pub_P;
        }
        else if (!P_buffer.empty())
        {   
            // std::cout << "P_buffer size: " << P_buffer.size() << std::endl;
            if (P_buffer.size() == 1)
            {
                Eigen::Matrix<float, 3, 1> pub_P = P_buffer[0];
                // if((last_pub_P.norm() < 0.01 || (last_pub_P - pub_P).norm()< 0.6) && pub_P.norm() > 0.2)
                // {
                    gap_pose.pose.position.x = pub_P[0];
                    gap_pose.pose.position.y = pub_P[1];
                    gap_pose.pose.position.z = pub_P[2];
                    gap_array.poses.push_back(gap_pose.pose);
                    gap_array.header.stamp = ros::Time::now();
                    gap_pose_pub.publish(gap_array); 
                    gap_array.poses.clear();
                    last_pub_P = pub_P;
                // }
            }
            if(P_buffer.size() == 2)
            {
                Eigen::Matrix<float, 3, 1> pub_P1 = P_buffer[0];
                Eigen::Matrix<float, 3, 1> pub_P2 = P_buffer[1];
                int i;
                int j;
                if (P_buffer[1][1] > P_buffer[0][1])
                {
                    i = 0;
                    j = 1;
                }
                else
                {
                    i = 1;
                    j = 0;
                }
                if ((last_pub_P.norm() < 0.01 || (last_pub_P - P_buffer[i]).norm()< 0.6) && (last_pub_P2.norm() < 0.01 || (last_pub_P2 - P_buffer[j]).norm()< 0.6) && P_buffer[i].norm() > 0.5 && P_buffer[j].norm() > 0.5)
                {    
                    gap_pose.pose.position.x = P_buffer[i][0];
                    gap_pose.pose.position.y = P_buffer[i][1];
                    gap_pose.pose.position.z = P_buffer[i][2];
                    gap_array.poses.push_back(gap_pose.pose);
                    gap_pose.pose.position.x = P_buffer[j][0];
                    gap_pose.pose.position.y = P_buffer[j][1];
                    gap_pose.pose.position.z = P_buffer[j][2];
                    gap_array.poses.push_back(gap_pose.pose);
                    gap_array.header.stamp = ros::Time::now();
                    gap_pose_pub.publish(gap_array); 
                    gap_array.poses.clear();
                    last_pub_P = P_buffer[i];
                    last_pub_P2 = P_buffer[j];
                }
            }
            P_buffer.clear();
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
