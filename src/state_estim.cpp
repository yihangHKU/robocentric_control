#include <ros/ros.h>
#include "state_estim.hpp"

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "state_estim");
    ros::NodeHandle nh;
    const int process_noise_dof = 12;
    const int measurement_noise_dof = 6;
    const int Maximum_iter = 5;

    state init_state;
    esekfom::esekf<state, process_noise_dof, input, measurement, measurement_noise_dof>::cov init_P;
    esekfom::esekf<state, process_noise_dof, input, measurement, measurement_noise_dof> kf(init_state, init_P);

    input i;

    // std::cout << "rot: " << state:: << std::endl;
    
    ros::spin();    
}