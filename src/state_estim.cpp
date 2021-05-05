#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <deque>
#include "state_estim.hpp"
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>


std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
std::deque<geometry_msgs::PoseStamped::ConstPtr> gap_buffer;
std::deque<double> log_dt;
std::deque<state> log_state;
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

void gap_cb(const geometry_msgs::PoseStamped::ConstPtr &msg_in)
{
    geometry_msgs::PoseStamped::Ptr msg(new geometry_msgs::PoseStamped(*msg_in));
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
    ros::Publisher grav_pub = nh.advertise<geometry_msgs::Vector3Stamped> ("robocentric/gravity", 100);
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Vector3Stamped> ("robocentric/velocity", 100);
    geometry_msgs::PoseStamped pose;
    geometry_msgs::Vector3Stamped grav;
    geometry_msgs::Vector3Stamped vel;
    double step = 0;
    ros::Rate rate(3000);
    const int process_noise_dof = 12;
    const int measurement_noise_dof = 6;
    int Maximum_iter = 1;
    state::scalar limit[state::DOF] = {0.0001};
    Eigen::Matrix<state::scalar, process_noise_dof, process_noise_dof> Q = 1. * Eigen::MatrixXd::Identity(process_noise_dof, process_noise_dof);
    for (int i = 6; i<process_noise_dof; i++)
    {
        Q(i,i) = 0.0001;
    }
    Eigen::Matrix<state::scalar, measurement_noise_dof, measurement_noise_dof> R = 0.01 * Eigen::MatrixXd::Identity(measurement_noise_dof, measurement_noise_dof);
    ros::spinOnce();
    state init_state;
    esekfom::esekf<state, process_noise_dof, input, measurement, measurement_noise_dof>::cov init_P = 1 * Eigen::MatrixXd::Identity(state::DOF, state::DOF);
    for (int i = 17; i < state::DOF; i++)
    {
        init_P(i,i) = 0.001;
    }
    esekfom::esekf<state, process_noise_dof, input, measurement, measurement_noise_dof> kf;
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
                Eigen::Matrix<double, 3, 1> p_c_r;
                p_c_r << gap_buffer.front()->pose.position.x, gap_buffer.front()->pose.position.y, gap_buffer.front()->pose.position.z;
                init_state.pos = init_state.offset_R_C_B.toRotationMatrix().inverse() * (p_c_r - init_state.offset_P_C_B);
                // init_state.pos = p_c_r - offset_P_C_B;
                Eigen::Quaternion<double> q_cr(gap_buffer.front()->pose.orientation.w, gap_buffer.front()->pose.orientation.x, gap_buffer.front()->pose.orientation.y, gap_buffer.front()->pose.orientation.z);
                init_state.rot = init_state.offset_R_C_B.toRotationMatrix().inverse() * q_cr.toRotationMatrix();
                // init_state.rot = q_cr;                
                // init_state.grav.vec = vect3(init_state.offset_R_C_B.toRotationMatrix().inverse() * q_cr.toRotationMatrix() * init_state.grav.vec);
                // init_state.grav.vec = vect3(q_cr.toRotationMatrix() * init_state.grav.vec);                
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
            Eigen::Vector3d euler_cur = SO3ToEuler(s_log.rot);
            Eigen::Vector3d euler_offset = SO3ToEuler(s_log.offset_R_C_B);
            Eigen::Vector3d grav_r = s_log.rot.toRotationMatrix().transpose() * s_log.grav.vec;
            Eigen::Vector3d pos_r = - s_log.rot.toRotationMatrix().transpose() * s_log.pos;
            fout_pre << step << " " << euler_cur.transpose() << " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
            << " " << s_log.bg.transpose() << " " << s_log.ba.transpose()<< " " << s_log.grav.vec.transpose() \
            << " " << s_log.offset_P_C_B.transpose() << " " << euler_offset.transpose() << " " << grav_r.transpose() <<  " " \
            << pos_r.transpose() << std::endl;
            // fout_pre << step << " " << euler_cur.transpose() << " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
            // << " " << s_log.grav.vec.transpose() << std::endl;
            gap_measure.R_cr.x() = gap_buffer.front()->pose.orientation.x;
            gap_measure.R_cr.y() = gap_buffer.front()->pose.orientation.y;
            gap_measure.R_cr.z() = gap_buffer.front()->pose.orientation.z;
            gap_measure.R_cr.w() = gap_buffer.front()->pose.orientation.w;
            gap_measure.P_cr[0] = gap_buffer.front()->pose.position.x;
            gap_measure.P_cr[1] = gap_buffer.front()->pose.position.y;
            gap_measure.P_cr[2] = gap_buffer.front()->pose.position.z;
            kf.update_iterated(gap_measure, R);
            s_log = kf.get_x();
            euler_cur = SO3ToEuler(s_log.rot);
            euler_offset = SO3ToEuler(s_log.offset_R_C_B);
            grav_r = s_log.rot.toRotationMatrix().transpose() * s_log.grav.vec;
            pos_r = - s_log.rot.toRotationMatrix().transpose() * s_log.pos;
            Eigen::Vector3d euler_mear = SO3ToEuler(gap_measure.R_cr);
            fout_out << step << " " << euler_cur.transpose() << " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
            << " " << s_log.bg.transpose() << " " << s_log.ba.transpose()<< " " << s_log.grav.vec.transpose() \
            << " " << s_log.offset_P_C_B.transpose() << " " << euler_offset.transpose() << " " << grav_r.transpose() \
            << " " << pos_r.transpose() << " " << euler_mear.transpose() << " " <<  gap_measure.P_cr.transpose() << std::endl;
            // fout_out << step << " " << euler_cur.transpose() << " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
            // << " " << s_log.grav.vec.transpose() << " " << euler_mear.transpose() \
            // << " " <<  gap_measure.P_cr.transpose()<< std::endl;
            // std::cout << "update state: " << kf.get_x() << std::endl;
            log_state.push_back(kf.get_x());
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
            Eigen::Vector3d euler_cur = SO3ToEuler(s_log.rot);
            Eigen::Vector3d euler_offset = SO3ToEuler(s_log.offset_R_C_B);
            Eigen::Vector3d grav_r = s_log.rot.toRotationMatrix().transpose() * s_log.grav.vec;
            Eigen::Vector3d pos_r = - s_log.rot.toRotationMatrix().transpose() * s_log.pos;
            fout_pre << step << " " << euler_cur.transpose() << " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
            << " " << s_log.bg.transpose() << " " << s_log.ba.transpose()<< " " << s_log.grav.vec.transpose() \
            << " " << s_log.offset_P_C_B.transpose() << " " << euler_offset.transpose() << " " << grav_r.transpose() << " " \
            << pos_r.transpose() << std::endl;
            // fout_pre << step << " " << euler_cur.transpose() << " " << s_log.pos.transpose() << " " << s_log.vel.transpose() \
            // << " " << s_log.grav.vec.transpose() << std::endl;
            fout_input << step << " " << imu_input.omega.transpose() << " " << imu_input.a.transpose() << std::endl;
            // std::cout << "P prior: ";
            // for (int i = 0; i < state::DOF; i++)
            // {
            //     std::cout << " " << kf.get_P()(i,i);
            //     P_diag(i) = kf.get_P()(i,i);
            // }
            // std::cout << std::endl;
            log_state.push_back(kf.get_x());
            last_predict_time = imu_buffer.front()->header.stamp.toSec();
            imu_buffer.pop_front();
        }      
        mtx_buffer.unlock(); 
        double pub_time_now = ros::Time::now().toSec();
        if (flg_state_init && (pub_time_now - pub_time_last >= 0.01))
        {   
            state s = kf.get_x();
            pose.header.stamp = ros::Time::now();
            pose.pose.position.x = s.pos[0];
            pose.pose.position.y = s.pos[1];
            pose.pose.position.z = s.pos[2];
            if (DETECTION_MODE == GAP)
            {
                pose.pose.orientation.x = s.rot.x();
                pose.pose.orientation.y = s.rot.y();
                pose.pose.orientation.z = s.rot.z();
                pose.pose.orientation.w = s.rot.w();
            }
            pose_pub.publish(pose);
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