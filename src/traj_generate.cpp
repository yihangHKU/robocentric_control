#include "../include/poly_traj_tuils/poly_traj_utils/include/poly_traj_utils/am_traj_plus.hpp"
#include "../include/poly_traj_tuils/poly_traj_utils/include/poly_traj_utils/am_traj.hpp"
#include "../include/poly_traj_tuils/poly_traj_utils/include/poly_traj_utils/poly_visual_utils.hpp"
#include <iostream>
#include <fstream>

// Creat the shared_ptr
order7::AmTraj::Ptr am_ptr_7;
order5::AmTraj::Ptr am_ptr_5;
int main(int argc, char* argv[])
{   
    std::ofstream fout_traj;
    fout_traj.open("/home/dji/catkin_ws/src/robocentric_control/traj.txt", std::ios::out);
    // Reset memory
    am_ptr_5.reset(new order5::AmTraj);
    am_ptr_7.reset(new order7::AmTraj);
    // Init parameters
    float weight_T =  2024.0;
    float weight_acc = 0.1;
    float weight_jerk = 0.1;
    float weight_snap = 1;
    float max_v = 3;
    float max_a = 3;
    int max_it = 23;
    float eps = 0.02; 
    am_ptr_7->init(weight_T, weight_acc, weight_jerk, weight_snap, max_v, max_a, max_it, eps);
    // am_ptr_5->init(op_5.weight_T, op_5.weight_acc,
    //             op_5.weight_jerk, op_5.max_v, op_5.max_a, op_5.max_it, op_5.eps);

    Trajectory traj_5,traj_7;
    // Generate constrained opt_traj
    std::vector<Eigen::Vector3d> waypts;
    waypts.push_back(Vec3(-1.777,-0.223,0.9865));
    waypts.push_back(Vec3(0,0,0));
    traj_7=  am_ptr_7->genOptimalTrajDTC(waypts, Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0.6, 0, 0), Vec3(0, 8.487, -4.90), Vec3(0, 0, 0));
    std::cout << "trajectory: " << traj_7.getPos(0.5) << std::endl;
    std::cout << "durations: " << traj_7.getTotalDuration() << std::endl;
    float sample_time = 0.01;
    for (int i = 0; i < int(traj_7.getTotalDuration() / 0.01) + 1; i++)
    {
        fout_traj << i * sample_time << " " << traj_7.getPos(i * sample_time).transpose() << std::endl;
    }
    return 0;
}