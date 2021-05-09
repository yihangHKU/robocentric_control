#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <deque>
#include "state_estim_gap.hpp"
// #include "state_estim_hover.hpp"
//#include "state_estim_hover2.hpp"
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Vector3Stamped.h>

std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<geometry_msgs::PoseArray::ConstPtr> gap_buffer;
std::deque<double> log_dt;
std::deque<Eigen::Matrix<double, state::DOF, 1>> log_P_prior;
std::mutex mtx_buffer;
double last_timestamp_gap = -1;
double last_timestamp_imu   = -1;


void imu_cb(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
    }
    last_timestamp_imu = msg->header.stamp.toSec();
    imu_buffer.push_back(msg);
    // std::cout << "imu buffer size: " << imu_buffer.size() << std::endl;
    // std::cout << "imu time: " << last_timestamp_imu - 1.61693e+09 << std::endl;
    mtx_buffer.unlock();
}

void gap_cb(const geometry_msgs::PoseArray::ConstPtr &msg_in)
{
    geometry_msgs::PoseArray::Ptr msg(new geometry_msgs::PoseArray(*msg_in));
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_timestamp_gap)
    {
        ROS_ERROR("gap loop back, clear buffer");
        gap_buffer.clear();
    }
    last_timestamp_gap = msg->header.stamp.toSec();
    gap_buffer.push_back(msg);
    // std::cout << "gap buffer size: " << gap_buffer.size() << std::endl;
    // std::cout << "gap time: " << last_timestamp_gap - 1.61693e+09 << std::endl;
    mtx_buffer.unlock();
}

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "state_estim");
    ros::NodeHandle nh;
    std::ofstream fout_pre, fout_out, fout_input;
    fout_pre.open("/home/dji/catkin_ws/debug/mat_pre.txt", std::ios::out);
    fout_out.open("/home/dji/catkin_ws/debug/mat_out.txt", std::ios::out);
    fout_input.open("/home/dji/catkin_ws/debug/imu.txt", std::ios::out);
    ros::Subscriber imu_sub = nh.subscribe("/mavros/imu/data_raw", 1000, imu_cb);
    ros::Subscriber gap_sub = nh.subscribe("/robocentric/camera/gap_pose", 1000, gap_cb);
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped> ("robocentric/pose", 100);
    ros::Publisher point_pub = nh.advertise<geometry_msgs::PointStamped> ("robocentric/point2", 100);
    ros::Publisher grav_pub = nh.advertise<geometry_msgs::Vector3Stamped> ("robocentric/gravity", 100);
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Vector3Stamped> ("robocentric/velocity", 100);
    geometry_msgs::PoseStamped pose;
    geometry_msgs::PointStamped point2;
    geometry_msgs::Vector3Stamped grav;
    geometry_msgs::Vector3Stamped vel;
    double step = 0;
    ros::Rate rate(3000);
    const int process_noise_dof = 12;
    int Maximum_iter = 1;
    state::scalar limit[state::DOF] = {0.0001};
    Eigen::Matrix<state::scalar, process_noise_dof, process_noise_dof> Q = 1. * Eigen::MatrixXd::Identity(process_noise_dof, process_noise_dof);
    for (int i = 6; i<process_noise_dof; i++)
    {
        Q(i,i) = 0.0001;
    }
    Eigen::Matrix<state::scalar, measurement::DOF, measurement::DOF> R = 0.01 * Eigen::MatrixXd::Identity(measurement::DOF, measurement::DOF);
    ros::spinOnce();
    state init_state;
    esekfom::esekf<state, process_noise_dof, input, measurement, measurement::DOF>::cov init_P = 1 * Eigen::MatrixXd::Identity(state::DOF, state::DOF);
    for (int i = state::DOF - 6; i < state::DOF; i++)
    {
        init_P(i,i) = 0.001;
    }
    esekfom::esekf<state, process_noise_dof, input, measurement, measurement::DOF> kf;
    double last_predict_time = -1;
    double last_update_time = -1;
    double dt = 0;
    bool flg_predict_init = false;
    bool flg_state_init = false;
    double pub_time_last = 0.0;
    input imu_input;
    measurement gap_measure;
    Eigen::Quaternionf q;
    Eigen::Matrix<double, state::DOF, 1> P_diag;
    while(ros::ok())
    {   
        ros::spinOnce();
        mtx_buffer.lock();
        if(!flg_state_init)
        {   
            if(!gap_buffer.empty())
            {
                init_state.vel[0] = 0;
                init_state.vel[1] = 0;
                init_state.vel[2] = 0;
                init_state.grav.vec[0] = 0;
                init_state.grav.vec[1] = 0;
                init_state.grav.vec[2] = 9.8;
                init_state.offset_R_C_B.x() = 0;
                init_state.offset_R_C_B.y() = 0;
                init_state.offset_R_C_B.z() = 0;
                init_state.offset_R_C_B.w() = 1;
                init_state.offset_P_C_B[0] = -0.05;
                init_state.offset_P_C_B[1] = 0;
                init_state.offset_P_C_B[2] = 0.09;
                vect3 offset_P_C_B;
                offset_P_C_B[0] = -0.05;
                offset_P_C_B[1] = 0;
                offset_P_C_B[2] = 0.11;
                state_init(init_state, gap_buffer.front());               
                kf.change_x(init_state);
                kf.change_P(init_P);
                kf.init(f, df_dx, df_dw, h, dh_dx, dh_dv, Maximum_iter, limit);
                flg_state_init = true;
                std::cout << "init state: " << kf.get_x() << std::endl;
                // std::cout << "init P: " << kf.get_P() << std::endl;
                gap_buffer.pop_front();
            }
            else if(!imu_buffer.empty())
            {
                imu_buffer.pop_front();
            }
        }

        if (flg_state_init  && !flg_predict_init && !imu_buffer.empty())
        {
            last_predict_time = imu_buffer.front()->header.stamp.toSec();
            flg_predict_init = true;
            imu_input.a[0] = imu_buffer.front()->linear_acceleration.x;
            imu_input.a[1] = - imu_buffer.front()->linear_acceleration.y;
            imu_input.a[2] = - imu_buffer.front()->linear_acceleration.z;
            imu_input.omega[0] = imu_buffer.front()->angular_velocity.x;
            imu_input.omega[1] = - imu_buffer.front()->angular_velocity.y;
            imu_input.omega[2] = - imu_buffer.front()->angular_velocity.z;
            imu_buffer.pop_front();
        }
        

        if (flg_state_init && flg_predict_init && !gap_buffer.empty() && gap_buffer.front()->header.stamp.toSec() - last_update_time > 0)
        {
            dt = gap_buffer.front()->header.stamp.toSec() - last_predict_time;
            log_dt.push_back(dt);
            // std::cout << "dt1: " << dt << std::endl;
            kf.predict(dt, Q, imu_input);
            // std::cout << "predict state: " << kf.get_x() << std::endl;
            // std::cout << "P prior: ";
            // for (int i = 0; i < state::DOF; i++)
            // {
            //     std::cout << " " << kf.get_P()(i,i);
            //     P_diag(i) = kf.get_P()(i,i);
            // }
            // std::cout << std::endl;
            step += dt;
            state s_log = kf.get_x();
            predict_log(fout_pre, s_log, step); 
            measure_receive(gap_measure, gap_buffer.front());
            kf.update_iterated(gap_measure, R);
            s_log = kf.get_x();
            update_log(fout_out, s_log, gap_measure, step);
            last_predict_time = gap_buffer.front()->header.stamp.toSec();
            last_update_time = gap_buffer.front()->header.stamp.toSec();
            gap_buffer.pop_front();
        }
        while (!imu_buffer.empty() && imu_buffer.front()->header.stamp.toSec() - last_predict_time < 0)
        {
            std::cout << "delta t: " << imu_buffer.front()->header.stamp.toSec() - last_predict_time << std::endl;
            imu_buffer.pop_front();
        }
        if(flg_state_init && flg_predict_init && !imu_buffer.empty() && imu_buffer.front()->header.stamp.toSec() - last_predict_time > 0)
        {
            dt = imu_buffer.front()->header.stamp.toSec() - last_predict_time;
            log_dt.push_back(dt);
            // std::cout << "dt2: " << dt << std::endl;
            imu_input.a[0] = imu_buffer.front()->linear_acceleration.x;
            imu_input.a[1] = - imu_buffer.front()->linear_acceleration.y;
            imu_input.a[2] = - imu_buffer.front()->linear_acceleration.z;
            imu_input.omega[0] = imu_buffer.front()->angular_velocity.x;
            imu_input.omega[1] = - imu_buffer.front()->angular_velocity.y;
            imu_input.omega[2] = - imu_buffer.front()->angular_velocity.z;
            kf.predict(dt, Q, imu_input);
            step += dt;
            state s_log = kf.get_x();
            predict_log(fout_pre, s_log, step); 
            fout_input << step << " " << imu_input.omega.transpose() << " " << imu_input.a.transpose() << std::endl;
            // std::cout << "P prior: ";
            // for (int i = 0; i < state::DOF; i++)
            // {
            //     std::cout << " " << kf.get_P()(i,i);
            //     P_diag(i) = kf.get_P()(i,i);
            // }
            // std::cout << std::endl;
            last_predict_time = imu_buffer.front()->header.stamp.toSec();
            imu_buffer.pop_front();
        }      
        mtx_buffer.unlock(); 
        double pub_time_now = ros::Time::now().toSec();
        if (flg_state_init && (pub_time_now - pub_time_last >= 0.01))
        {   
            state s = kf.get_x();
            pose.header.stamp = ros::Time::now();
            point2.header.stamp = ros::Time::now();
            topic_pub(s, pose, point2);
            pose_pub.publish(pose);
            point_pub.publish(point2);
            grav.header.stamp = ros::Time::now();
            grav.vector.x = s.grav.vec[0];
            grav.vector.y = s.grav.vec[1];
            grav.vector.z = s.grav.vec[2];
            grav_pub.publish(grav);
            vel.header.stamp = ros::Time::now();
            vel.vector.x = s.vel[0];
            vel.vector.y = s.vel[1];
            vel.vector.z = s.vel[2];
            vel_pub.publish(vel);
            pub_time_last = pub_time_now;
        }
        rate.sleep();
    }
    ros::spin(); 
    return 0;
}
