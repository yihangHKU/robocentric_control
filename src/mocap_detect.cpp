#include <ros/ros.h>
#include <iostream>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include "vision_detect.hpp"
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>

geometry_msgs::PoseStamped gap_pose;
geometry_msgs::PoseStamped aircraft_pose;
geometry_msgs::PoseStamped corner1;
geometry_msgs::PoseStamped corner2;
geometry_msgs::PoseStamped corner3;
geometry_msgs::PoseStamped corner4;
sensor_msgs::Imu Imu;
Eigen::Quaternionf q_aircraft;


void corner1_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    corner1 = *msg;
}

void corner2_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    corner2 = *msg;
}

void corner3_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    corner3 = *msg;
}

void corner4_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    corner4 = *msg;
}

void aircraft_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    aircraft_pose = *msg;
}

void aircraft_q_cb(const sensor_msgs::Imu::ConstPtr &msg)
{   
    Imu = *msg;
    q_aircraft.x() = (float)Imu.orientation.x;
    q_aircraft.y() = (float)Imu.orientation.y;
    q_aircraft.z() = (float)Imu.orientation.z;
    q_aircraft.w() = (float)Imu.orientation.w;
    // std::cout << "q_aircraft: " << q_aircraft.x() << " " << q_aircraft.y() << " " << q_aircraft.z() << " " << q_aircraft.w() << std::endl;
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "mocap_detect");
    ros::NodeHandle nh;
    
    ros::Publisher gap_pose_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("/robocentric/camera/gap_pose", 100);
    ros::Publisher vision_pose_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("/mavros/vision_pose/pose", 100);
    ros::Subscriber corner1_sub = nh.subscribe("/corner1/viconros/mocap/pos", 1000, corner1_cb);
    ros::Subscriber corner2_sub = nh.subscribe("/corner2/viconros/mocap/pos", 1000, corner2_cb);
    ros::Subscriber corner3_sub = nh.subscribe("/corner3/viconros/mocap/pos", 1000, corner3_cb);
    ros::Subscriber corner4_sub = nh.subscribe("/corner4/viconros/mocap/pos", 1000, corner4_cb);
    ros::Subscriber aircraft_sub = nh.subscribe("/aircraft/viconros/mocap/pos", 1000, aircraft_cb);
    ros::Subscriber aircraft_q_sub = nh.subscribe("/mavros/imu/data", 1000, aircraft_q_cb);
    ros::Rate rate(30.0);
    for(int i = 0; i < 10; i++)
    {
        ros::spinOnce();
        rate.sleep();
    }

    while (ros::ok()){
        ros::spinOnce();
        vector<Point3f> points(0);
        points.push_back(Point3f(corner1.pose.position.x, corner1.pose.position.y, corner1.pose.position.z));
        points.push_back(Point3f(corner2.pose.position.x, corner2.pose.position.y, corner2.pose.position.z));
        points.push_back(Point3f(corner3.pose.position.x, corner3.pose.position.y, corner3.pose.position.z));
        points.push_back(Point3f(corner4.pose.position.x, corner4.pose.position.y, corner4.pose.position.z));
        vector<float> plane_norm = {0, 0, 0};
        vector<float> centroid = {0, 0, 0};
        plane_from_points(points, plane_norm, centroid);
        // std::cout << "gap center: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;
        vector<float> gap_direction_x = {0.,0.,0.};
        vector<float> gap_direction_z = {-plane_norm[0], -plane_norm[1], -plane_norm[2]};
        gap_direction_x[0] = 0.5 * (points[1].x + points[2].x - points[0].x - points[3].x);
        gap_direction_x[1] = 0.5 * (points[1].y + points[2].y - points[0].y - points[3].y);
        gap_direction_x[2] = 0.5 * (points[1].z + points[2].z - points[0].z - points[3].z);
        vector<float> gap_direction_y = vector_product(gap_direction_z, gap_direction_x);
        float length_ = norm(gap_direction_y);
        for (int i = 0; i<3; i++)
        {   
            gap_direction_y[i] = float(gap_direction_y[i] / length_);
        }
        gap_direction_x = vector_product(gap_direction_y, gap_direction_z);
        Eigen::Matrix<float, 3, 3> R_gap;
        R_gap << gap_direction_z[0], -gap_direction_x[0], -gap_direction_y[0],
            gap_direction_z[1], -gap_direction_x[1], -gap_direction_y[1],
            gap_direction_z[2], -gap_direction_x[2], -gap_direction_y[2];
        // std::cout << "R_gap" << R_gap << std::endl;
	    Eigen::Quaternionf q_gap(R_gap);        
	//std::cout << "q_gap: " << q_gap.x() << " " << q_gap.y() << " " << q_gap.z() << " " << q_gap.w() << std::endl;
	    Eigen::Matrix<float, 3, 3> R_aircraft;
        R_aircraft = q_aircraft.toRotationMatrix();
        R_aircraft.block(0,0,3,1) = R_aircraft.block(0,0,3,1);
        R_aircraft.block(0,1,3,1) =  R_aircraft.block(0,1,3,1);
        R_aircraft.block(0,2,3,1) =  R_aircraft.block(0,2,3,1);
        Eigen::Matrix<float, 3, 3> R_b_r = R_aircraft.transpose() * R_gap;
        // std::cout << "R_b_r: " << R_b_r.block(0,0,3,1) << std::endl;
	    Eigen::Matrix<float, 3, 1> P_b_r = {centroid[0] - (float)aircraft_pose.pose.position.x, centroid[1] - (float)aircraft_pose.pose.position.y, centroid[2] - (float)aircraft_pose.pose.position.z};
         //std::cout << "P_b_r " << P_b_r << std::endl;
        //std::cout << "R_aircraft: " << R_aircraft << std::endl;
        P_b_r = R_aircraft.transpose() * P_b_r;
        //std::cout << "R_b_r " << R_b_r << std::endl;
        R_aircraft.block(0,0,3,1) = ENU_2_NED(R_aircraft.block(0,0,3,1));
        R_aircraft.block(0,1,3,1) = -ENU_2_NED(R_aircraft.block(0,1,3,1));
        R_aircraft.block(0,2,3,1) = -ENU_2_NED(R_aircraft.block(0,2,3,1));
	    R_gap.block(0,0,3,1) = ENU_2_NED(R_gap.block(0,0,3,1));
	    R_gap.block(0,1,3,1) = -ENU_2_NED(R_gap.block(0,1,3,1));
	    R_gap.block(0,2,3,1) = -ENU_2_NED(R_gap.block(0,2,3,1));
	    R_b_r = R_aircraft.transpose() * R_gap;	    
	    P_b_r = FLU_2_FRD(P_b_r);
        Eigen::Quaternionf q_b_r(R_b_r);
        geometry_msgs::PoseStamped gap_pose;
        gap_pose.header.stamp = ros::Time::now();
        gap_pose.header.frame_id = "FRD";
        gap_pose.pose.position.x = P_b_r[0];
        gap_pose.pose.position.y = P_b_r[1];
        gap_pose.pose.position.z = P_b_r[2];
        gap_pose.pose.orientation.x = q_b_r.x();
        gap_pose.pose.orientation.y = q_b_r.y();
        gap_pose.pose.orientation.z = q_b_r.z();
        gap_pose.pose.orientation.w = q_b_r.w();
        gap_pose_pub.publish(gap_pose);
        vision_pose_pub.publish(aircraft_pose);
        rate.sleep();
    }
    return 0;
}