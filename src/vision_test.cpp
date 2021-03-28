#include <ros/ros.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
using namespace cv; 
using namespace std;

sensor_msgs::Image color_frame;
sensor_msgs::Image depth_frame;
sensor_msgs::CameraInfo color_info;

void color_frame_cb(const sensor_msgs::Image::ConstPtr& msg)
{
    color_frame = *msg;
}   

void depth_frame_cb(const sensor_msgs::Image::ConstPtr& msg)
{
    depth_frame = *msg;
}   

void color_info_cb(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    color_info = *msg;
}

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "state_test");
    ros::NodeHandle nh;
    ros::Rate rate(10.0);

    ros::Subscriber color_frame_sub = nh.subscribe<sensor_msgs::Image>
        ("/camera/color/camera_raw", 10, color_frame_cb);
    ros::Subscriber depth_frame_sub = nh.subscribe<sensor_msgs::Image>
        ("/camera/depth/camera_raw", 10, depth_frame_cb);
    ros::Subscriber color_info_sub = nh.subscribe<sensor_msgs::CameraInfo>
        ("/camera/color/camera_info", 10, color_info_cb);

    
    for(int i = 0; i < 30; i++)
    {
        ros::spinOnce();
        rate.sleep();
    }

    // cout << "color data: " << color_frame.data[0] << endl;
    cout << "color encoding: " << color_frame.encoding << endl;
    // cout << "color info: " << color_info. << endl;
    cout << "depth data: " << sizeof(depth_frame.data) << endl;
    cout << "color step: " << color_frame.step << endl;

    ros::spin();
}