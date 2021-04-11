#include <ros/ros.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "vision_detect.hpp" 
using namespace cv; 
using namespace std;

cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImagePtr cv_ptr_depth;
Mat color(Size(848, 480), CV_8UC3);
Mat depth(Size(848, 480), CV_16UC1);
Mat depth_new;
static const std::string OPENCV_WINDOW = "Image window";
void color_cb(const sensor_msgs::Image::ConstPtr &msg)
{   
    try
    {
       cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    color = cv_ptr->image;
}
void depth_cb(const sensor_msgs::Image::ConstPtr &msg)
{   
    try
    {
       cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO16);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    depth = cv_ptr->image;
    depth.convertTo(depth_new, CV_8UC1, 0.1, 0); 
}

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "cv_bridge_test");
    ros::NodeHandle nh;
    ros::Publisher color_pub = nh.advertise<sensor_msgs::Image>
            ("camera/color_raw", 100);
    ros::Subscriber color_sub = nh.subscribe("/camera/color/image_raw", 1000, color_cb);
    ros::Rate rate(10.0);
    std::cout << "color once " << std::endl;
    sensor_msgs::Image img_msg;
    std_msgs::Header header;
    cv_bridge::CvImage img_bridge;
    for (int i = 0; i < 10; i++)
    {
        ros::spinOnce();
    }
    // Mat color 
    while(ros::ok()){ 
        ros::spinOnce();
        // imshow(OPENCV_WINDOW, color);
        // waitKey(0);
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, color);
        img_bridge.toImageMsg(img_msg);
        color_pub.publish(img_msg);
        rate.sleep();
    }
    
    ros::spin();
    return 0;
}