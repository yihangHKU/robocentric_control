cmake_minimum_required(VERSION 3.0.2)
project(robocentric_control)
add_compile_options(-std=c++11)
# set(cv_brige_DIR /opt/ros/kinetic/share/cv_bridge/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(poly_traj_utils)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robocentric_control
  CATKIN_DEPENDS roscpp rospy std_msgs poly_traj_utils cv_bridge
)

include_directories(
 include
  ${realsense2_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

#add_library(EKF
#include/IKFoM_toolkit/eskfom/esefom.hpp
#)

add_executable(state_estim src/state_estim.cpp)
target_link_libraries(state_estim ${catkin_LIBRARIES})
add_executable(vision_detect src/vision_detect.cpp)
target_link_libraries(vision_detect ${catkin_LIBRARIES} ${realsense2_LIBRARY} ${OpenCV_LIBS})
#add_executable(traj_generate src/traj_generate.cpp)
#target_link_libraries(traj_generate ${catkin_LIBRARIES})
add_executable(vision_test src/vision_test.cpp)
target_link_libraries(vision_test ${catkin_LIBRARIES} ${realsense2_LIBRARY} ${OpenCV_LIBS})
add_executable(cv_bridge_test src/cv_bridge_test.cpp)
target_link_libraries(cv_bridge_test ${catkin_LIBRARIES} ${realsense2_LIBRARY} ${OpenCV_LIBS})
