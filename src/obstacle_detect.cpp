#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <deque>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/TwistStamped.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

std::deque<geometry_msgs::PoseArray::ConstPtr> ball_buffer;
Eigen::Matrix<float, 3, 1> vicon_pos;
Eigen::Matrix<float, 3, 1> vicon_vel;
double last_timestamp_gap = -1;
double last_ball_time = 0;
bool time_initial = false;
Eigen::Matrix<float, 3, 1> last_ball{0.0, 0.0, 0.0};
Eigen::Matrix<float, 3, 1> now_ball{0.0, 0.0, 0.0};
Eigen::Matrix<float, 3, 1> velocity{0.0, 0.0, 0.0};

void ball_cb(const geometry_msgs::PoseArray::ConstPtr &msg_in)
{
    geometry_msgs::PoseArray::Ptr msg(new geometry_msgs::PoseArray(*msg_in));
    if (msg->header.stamp.toSec() < last_timestamp_gap)
    {
        ROS_ERROR("gap loop back, clear buffer");
        ball_buffer.clear();
    }
    last_timestamp_gap = msg->header.stamp.toSec();
    ball_buffer.push_back(msg);
    // std::cout << "gap buffer size: " << gap_buffer.size() << std::endl;
    // std::cout << "gap time: " << last_timestamp_gap - 1.61693e+09 << std::endl;
}

void vicon_pos_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    vicon_pos << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
}

void vicon_vel_cb(const geometry_msgs::TwistStamped::ConstPtr &msg)
{
    vicon_vel << msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z;
}

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "obstacle_detect");
    ros::NodeHandle nh;
    std::ofstream fout_obs;
    ros::Subscriber ball_sub = nh.subscribe("/robocentric/camera/gap_pose2", 1000, ball_cb);
    ros::Subscriber vicon_pos_sub = nh.subscribe("/robocentric/vicon/ball_pos", 1000, vicon_pos_cb);
    ros::Subscriber vicon_vel_sub = nh.subscribe("/robocentric/vicon/ball_vel", 1000, vicon_vel_cb);
    fout_obs.open("/home/dji/catkin_ws/debug/mat_obs.txt", std::ios::out);
    ros::Rate rate(1000);
    double time = 0;
    while(ros::ok())
    {
        ros::spinOnce();

        if (!ball_buffer.empty())
        {   
            if (time_initial)
            {   
                now_ball << ball_buffer.front()->poses[0].position.x,  ball_buffer.front()->poses[0].position.y,  ball_buffer.front()->poses[0].position.z;
                double dt = ball_buffer.front()->header.stamp.toSec() - last_ball_time;
                if((now_ball - last_ball).norm() < 1.5 && dt < 0.5)
                {
                    velocity = (now_ball - last_ball) / dt;
                    fout_obs << time << " " << now_ball.transpose() << " " << velocity.transpose() << " " << vicon_pos.transpose() << " " << vicon_vel.transpose() << std::endl;   
                }
                else
                {   
                    std::cout << "delta p: " << (now_ball - last_ball).norm() << std::endl;
                    std::cout << "delta t: " << ball_buffer.front()->header.stamp.toSec() - last_ball_time << std::endl;
                }
                time += dt;
                last_ball = now_ball;
                last_ball_time = ball_buffer.front()->header.stamp.toSec();
            }
            else
            {
                last_ball = now_ball;
                last_ball_time = ball_buffer.front()->header.stamp.toSec();
                time_initial = true;
            }
            ball_buffer.clear();
        }

        rate.sleep();
    }
    ros::spin();
    return 0;
}