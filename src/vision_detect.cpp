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
int radius = 4;
RNG rng(12345);
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
Mat color_, gray_;
vector<Point2f> corners_;
float centroid_[3] = {0, 0, 0};
vector<float> direction_ = {0, 0, 0};
vector<Eigen::Matrix<float, 3, 1>> P_buffer;
vector<Eigen::Matrix<float, 3, 1>> P2_buffer;
vector<Eigen::Quaternionf> q_buffer;
std::ofstream fout_vision;
std::ofstream fout_depth;
int verbose = 1;
int step = 0;

sensor_msgs::Image color_frame;
sensor_msgs::Image depth_frame;
sensor_msgs::CameraInfo color_info;
geometry_msgs::PoseArray gap_array;
geometry_msgs::Pose gap_pose;

void plane_from_points(vector<Point3f> points)
{   
    vector<float> sum = {0.0, 0.0, 0.0};
    for (int i = 0; i < points.size(); i++)
    {   
        sum[0] += points[i].x;
        sum[1] += points[i].y;
        sum[2] += points[i].z;
    }
    for (int i = 0; i < 3; i++)
    {
        centroid_[i] = sum[i] * (1.0 / int(points.size()));
    }
    float xx = 0.0;
    float xy = 0.0;
    float xz = 0.0;
    float yy = 0.0;
    float yz = 0.0;
    float zz = 0.0;
    for (int i =0; i < points.size(); i++)
    {
        vector<float> r = {points[i].x - centroid_[0], points[i].y - centroid_[1], points[i].z - centroid_[2]};
        xx += r[0] * r[0];
        xy += r[0] * r[1];
        xz += r[0] * r[2];
        yy += r[1] * r[1];
        yz += r[1] * r[2];
        zz += r[2] * r[2];
    }
    
    float det_x = yy * zz - yz * yz;
    float det_y = xx * zz - xz * xz;
    float det_z = xx * yy - xy * xy;
    float det_max = max(max(det_x, det_y), det_z);
    vector<float> direction;
    if (det_max == det_x)
    {
        direction_[0] = det_x;
        direction_[1] = xz * yz - xy * zz;
        direction_[2] = xy * yz - xz * yy;
    }
    else if (det_max == det_y)
    {
        direction_[0] = xz * yz - xy * zz;
        direction_[1] = det_y;
        direction_[2] = xy * xz - yz * xx;
    }
    else
    {
        direction_[0] = xy * yz - xz * yy;
        direction_[1] = xy * xz - yz * xx;
        direction_[2] = det_z;
    }
    float length = norm(direction_);
    direction_[0] = direction_[0] / length;
    direction_[1] = direction_[1] / length;
    direction_[2] = direction_[2] / length;
    // cout << "plane centroid: ";
    // for (int i = 0; i < 3; i++)
    // {
    //     cout << " " << centroid_[i];
    // }
    // cout << endl;
    // cout << "plane norm direction: ";
    // for (int i = 0; i < 3; i++)
    // {
    //     cout << " " << direction_[i];
    // }
    // cout << endl;
}
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
    ros::Rate rate(50.0);
    // time_t t;
    // tm* local;
    // char buf[128] = {0};
    // t = time(NULL);
    // local = localtime(&t);
    // strftime(buf, 64, "%Y-%m-%d %H:%M:%S", local);
    fout_vision.open("/home/dji/catkin_ws/debug/mat_vision.txt", std::ios::out);
    fout_depth.open("/home/dji/catkin_ws/debug/mat_depth.txt", std::ios::out);
    Eigen::Matrix<float, 3, 1> last_pub_P{0.0, 0.0, 0.0};
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
        step++;
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
        Mat imgcanny_blur;
        cvtColor( color, gray, COLOR_BGR2GRAY );
        gray_ = gray;
        blur( gray, gray, Size(6,6) );
        Canny(gray, imgcanny_blur, 100, 200, 3, true);
        // namedWindow("imgCanny blur", CV_WINDOW_AUTOSIZE);
        // imshow("imgCanny blur", imgcanny_blur);
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours( imgcanny_blur, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        Mat imgcontour = Mat::zeros(imgcanny_blur.size(), CV_8UC3);
        list<int> rectangle_index;
        for (size_t i = 0; i < contours.size(); i++)
        {   
            approxPolyDP(Mat(contours[i]), contours[i], arcLength(contours[i], true) * 0.02, true);
            if (contours[i].size() == 4)
            {   
                rectangle_index.push_back(i);
                // float length_contour = arcLength(contours[i], false);
                // Scalar color = Scalar(100, 100, 100);
                // drawContours( imgcontour, contours, (int)i, color, 1, LINE_8, hierarchy, 0);
                // cout << "contour index: " << i << " length: " << length_contour << "hierarchy: " << hierarchy[i] << endl;
            }
        }
        list<int> gap_index;
        for (list<int>::iterator it = rectangle_index.begin(); it != rectangle_index.end(); it++)
        {
            if (hierarchy[*it][2] > 0 && hierarchy[hierarchy[*it][2]][2] > 0 
                && hierarchy[hierarchy[hierarchy[*it][2]][2]][2] > 0 && hierarchy[hierarchy[hierarchy[hierarchy[*it][2]][2]][2]][2] < 0)
            {   
                gap_index.push_back(*it);
                // cout << "gap index: " << *it << endl;
            }
        }
        bool depth_error = false;
        for (list<int>::iterator it = gap_index.begin(); it != gap_index.end(); it++)
        {
            Scalar color = Scalar(100, 100, 100);
            // drawContours( imgcontour, contours, (int)*it, color, 1, CV_FILLED, hierarchy, 0);
            int inner_contour_index = hierarchy[hierarchy[hierarchy[*it][2]][2]][2];
            // drawContours( imgcontour, contours, (int)inner_contour_index, color, 1, CV_FILLED, hierarchy, 0);
            // cout << "outer corners: " << contours[*it] << endl;
            // cout << "inner corners: " << contours[inner_contour_index] << endl;
            vector<Point2f> out_corners;
            vector<Point2f> in_corners;
            for (int i = 0; i < 4; i ++)
            {
                out_corners.push_back(contours[*it][i]);
                in_corners.push_back(contours[inner_contour_index][i]);
                circle( color_, out_corners[i], radius, Scalar(200, 0, 0), CV_FILLED );
                circle( color_, in_corners[i], radius, Scalar(200, 0, 0), CV_FILLED );
            }
            float out_points[4][3];
            float out_pixels[4][2];
            float in_points[4][3];
            float in_pixels[4][2];
            vector<Point3f> points_vec(0);
            vector<Point2f> pixels_vec(0);
            vector<float> out_depth_origin(0);
            vector<float> out_depth_search(0);
            vector<float> in_depth_origin(0);
            vector<float> in_depth_search(0);
            int range = 8;
            for (int i = 0; i < 4; i++)
            {   
                float out_corner_depth = depth_frame.get_distance(out_corners[i].x, out_corners[i].y);
                float in_corner_depth = depth_frame.get_distance(in_corners[i].x, in_corners[i].y);
                out_depth_origin.push_back(out_corner_depth);
                in_depth_origin.push_back(in_corner_depth);
                for (int j = -range; j < range+1; j++)
                {
                    for(int k = -range; k < range+1; k++)
                    {   
                        if (0 < out_corners[i].x + k && out_corners[i].x + k < width && 0 < out_corners[i].y + j && out_corners[i].y + j < height)
                        {
                            float depth_search = depth_frame.get_distance(out_corners[i].x + k, out_corners[i].y + j);
                            if(depth_search > 0.20 && (depth_search < out_corner_depth || out_corner_depth < 0.20))
                            {
                                out_corner_depth = depth_search;
                            }
                        }
                        if (0 < in_corners[i].x + k && in_corners[i].x + k < width && 0 <in_corners[i].y + j && in_corners[i].y + j < height)
                        {
                            float depth_search_in = depth_frame.get_distance(in_corners[i].x + k, in_corners[i].y + j);
                            if(depth_search_in > 0.20 && (depth_search_in < in_corner_depth || in_corner_depth < 0.20))
                            {
                                in_corner_depth = depth_search_in;
                            }
                        }
                    }
                }
                out_depth_search.push_back(out_corner_depth);
                in_depth_search.push_back(in_corner_depth);
                // cout << "corner pixels depth: " << depth_frame.get_distance(out_corners[i].x, out_corners[i].y) << endl;
                out_pixels[i][0] = out_corners[i].x;
                out_pixels[i][1] = out_corners[i].y;
                in_pixels[i][0] = in_corners[i].x;
                in_pixels[i][1] = in_corners[i].y;
                rs2_deproject_pixel_to_point(out_points[i], &depth_intrins, out_pixels[i], out_corner_depth);
                rs2_deproject_pixel_to_point(in_points[i], &depth_intrins, in_pixels[i], in_corner_depth);
                points_vec.push_back(Point3f(out_points[i][0], out_points[i][1], out_points[i][2]));
                points_vec.push_back(Point3f(in_points[i][0], in_points[i][1], in_points[i][2]));
                pixels_vec.push_back(Point2f(out_pixels[i][0], out_pixels[i][1]));
                pixels_vec.push_back(Point2f(in_pixels[i][0], in_pixels[i][1]));
            }
            plane_from_points(points_vec);
            cout << "publish time: " << ros::Time::now() << endl;
            float centroid_pixel[2] = {0, 0};
            rs2_project_point_to_pixel(centroid_pixel, &depth_intrins, centroid_);
            // cout << "centroid pixel: " << centroid_pixel[0] << " " << centroid_pixel[1] << endl;
            vector<Point2f> centroid_pixel_(1);
            centroid_pixel_[0] = Point2f(centroid_pixel[0], centroid_pixel[1]);
            circle( color_, centroid_pixel_[0], radius, Scalar(200, 0, 0), CV_FILLED );
            int left_index_o[2] = {10};
            int right_index_o[2] = {10};
            int left_up_index;
            int left_down_index;
            int right_up_index;
            int right_down_index;
            for (int i = 0; i < 4; i++)   // find out quadrangle right and left index
                {
                    if (out_corners[i].x < centroid_pixel[0])
                    {
                        if (left_index_o[0] == 10)
                        {
                            left_index_o[0] = i;
                        }
                        else
                        {
                            left_index_o[1] = i;
                        }
                    }
                    else
                    {
                        if (right_index_o[0] == 10)
                        {
                            right_index_o[0] = i;
                        }
                        else
                        {
                            right_index_o[1] = i; 
                        }
                    }
                }
            if (out_corners[left_index_o[0]].y < out_corners[left_index_o[1]].y) // find out quadrangle left_up_index and left_down_index
                {
                    left_up_index = left_index_o[0];
                    left_down_index = left_index_o[1];
                }
            else
                {
                    left_up_index = left_index_o[1];
                    left_down_index = left_index_o[0];
                }
            if (out_corners[right_index_o[0]].y < out_corners[right_index_o[1]].y) // find in quadrangle right_up_index and right_down_index
                {
                    right_up_index = right_index_o[0];
                    right_down_index = right_index_o[1];
                }
            else
                {
                    right_up_index = right_index_o[1];
                    right_down_index = right_index_o[0];
                }
            int out_quadr_index_[4] = {left_up_index, right_up_index, right_down_index, left_down_index};
            // cout << "out quadrangle index: " << out_quadr_index_[0] << ' ' << out_quadr_index_[1] << ' ' << out_quadr_index_[2] << ' ' << out_quadr_index_[3] << endl;
            int left_index_i[2] = {10};
            int right_index_i[2] = {10};
            for (int i = 0; i < 4; i++)   // find in quadrangle right and left index
                {
                    if (in_corners[i].x < centroid_pixel[0])
                    {
                        if (left_index_i[0] == 10)
                        {
                            left_index_i[0] = i;
                        }
                        else
                        {
                            left_index_i[1] = i;
                        }
                    }
                    else
                    {
                        if (right_index_i[0] == 10)
                        {
                            right_index_i[0] = i;
                        }
                        else
                        {
                            right_index_i[1] = i; 
                        }
                    }
                }
            if (in_corners[left_index_i[0]].y < in_corners[left_index_i[1]].y) // find in quadrangle left_up_index and left_down_index
                {
                    left_up_index = left_index_i[0];
                    left_down_index = left_index_i[1];
                }
            else
                {
                    left_up_index = left_index_i[1];
                    left_down_index = left_index_i[0];
                }
            if (in_corners[right_index_i[0]].y < in_corners[right_index_i[1]].y) // find in quadrangle right_up_index and right_down_index
                {
                    right_up_index = right_index_i[0];
                    right_down_index = right_index_i[1];
                }
            else
                {
                    right_up_index = right_index_i[1];
                    right_down_index = right_index_i[0];
                }
            int in_quadr_index_[4] = {left_up_index, right_up_index, right_down_index, left_down_index};
            fout_depth << step << " " << out_depth_origin[out_quadr_index_[0]] << " " << out_depth_origin[out_quadr_index_[1]] << " " << out_depth_origin[out_quadr_index_[2]] << " " << out_depth_origin[out_quadr_index_[3]] \
            << " " << in_depth_origin[in_quadr_index_[0]] << " " << in_depth_origin[in_quadr_index_[1]] << " " << in_depth_origin[in_quadr_index_[2]] << " " << in_depth_origin[in_quadr_index_[3]] << endl;
            fout_depth << step << " " << out_depth_search[out_quadr_index_[0]] << " " << out_depth_search[out_quadr_index_[1]] << " " << out_depth_search[out_quadr_index_[2]] << " " << out_depth_search[out_quadr_index_[3]] \
            << " " << in_depth_search[in_quadr_index_[0]] << " " << in_depth_search[in_quadr_index_[1]] << " " << in_depth_search[in_quadr_index_[2]] << " " << in_depth_search[in_quadr_index_[3]] << endl << endl;
            for (int i = 0; i < 4; i++)
            {
                if (abs(out_depth_search[out_quadr_index_[i]] - in_depth_search[in_quadr_index_[i]]) > 0.15)
                {
                    depth_error = true;
                }
            }
            float Point2[3];
            rs2_deproject_pixel_to_point(Point2, &depth_intrins, in_pixels[in_quadr_index_[1]], in_depth_search[in_quadr_index_[1]]);
            // cout << "in quadrangle index: " << in_quadr_index_[0] << ' ' << in_quadr_index_[1] << ' ' << in_quadr_index_[2] << ' ' << in_quadr_index_[3] << endl;
            vector<float> gap_direction_x = {0.,0.,0.};
            vector<float> gap_direction_z = {direction_[0], direction_[1], direction_[2]};
            for (int i = 0; i<3; i++)
                {
                    gap_direction_x[i] = 0.5 * (in_points[in_quadr_index_[1]][i] + in_points[in_quadr_index_[2]][i] - in_points[in_quadr_index_[0]][i] - in_points[in_quadr_index_[3]][i]);
                }
            vector<float> gap_direction_y = vector_product(gap_direction_z, gap_direction_x);
            float length_ = norm(gap_direction_y);
            for (int i = 0; i<3; i++)
            {   
                gap_direction_y[i] = float(gap_direction_y[i] / length_);
            }
            gap_direction_x = vector_product(gap_direction_y, gap_direction_z);
            // cout << "plane x direction: ";
            // for (int i = 0; i < 3; i++)
            // {
            //     cout << " " << gap_direction_x[i];
            // }
            // cout << endl;
            // cout << "plane y direction: ";
            // for (int i = 0; i < 3; i++)
            // {
            //     cout << " " << gap_direction_y[i];
            // }
            // cout << endl;
            vector<float> gap_center = {centroid_[0], centroid_[1], centroid_[2]};
            vector<float> point2_vector = {Point2[0], Point2[1], Point2[2]};
            point2_vector = RDF_2_FRD(point2_vector);
            gap_center = RDF_2_FRD(gap_center);
            gap_direction_x = RDF_2_FRD(gap_direction_x);
            gap_direction_y = RDF_2_FRD(gap_direction_y);
            gap_direction_z = RDF_2_FRD(gap_direction_z);
            if (verbose > 0)
            {
                Eigen::Matrix<float,1,3> gap_center_eigen(gap_center[0], gap_center[1], gap_center[2]);
                Eigen::Matrix<float,1,3> gap_direction_x_eigen(gap_direction_x[0], gap_direction_x[1], gap_direction_x[2]);
                Eigen::Matrix<float,1,3> gap_direction_y_eigen(gap_direction_y[0], gap_direction_y[1], gap_direction_y[2]);
                Eigen::Matrix<float,1,3> gap_direction_z_eigen(gap_direction_z[0], gap_direction_z[1], gap_direction_z[2]);
                fout_vision << step << " " << gap_center_eigen << " " << gap_direction_x_eigen \
                << " " << gap_direction_y_eigen << " " << gap_direction_z_eigen << std::endl; 
            }
            Eigen::Matrix<float, 3, 3> R;
            R << gap_direction_z[0], gap_direction_x[0], gap_direction_y[0],
                gap_direction_z[1], gap_direction_x[1], gap_direction_y[1],
                gap_direction_z[2], gap_direction_x[2], gap_direction_y[2];
            Eigen::Matrix<float, 3, 1> P;
            P << gap_center[0], gap_center[1], gap_center[2];
            Eigen::Matrix<float, 3, 1> P2;
            P2 << point2_vector[0], point2_vector[1], point2_vector[2];
            Eigen::Quaternionf q(R);
            P_buffer.push_back(P);
            P2_buffer.push_back(P2);
            q_buffer.push_back(q);
        }
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, color_);
        img_bridge.toImageMsg(img_msg);
        if (!depth_error)
        {
            color_pub.publish(img_msg);
        }
        if (!P_buffer.empty())
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
                gap_pose.position.x = pub_P[0];
                gap_pose.position.y = pub_P[1];
                gap_pose.position.z = pub_P[2];
                gap_pose.orientation.x = pub_q.x();
                gap_pose.orientation.y = pub_q.y();
                gap_pose.orientation.z = pub_q.z();
                gap_pose.orientation.w = pub_q.w();
                gap_array.poses.push_back(gap_pose);
                gap_pose.position.x = pub_P2[0];
                gap_pose.position.y = pub_P2[1];
                gap_pose.position.z = pub_P2[2];
                gap_array.poses.push_back(gap_pose);
                gap_array.header.stamp = ros::Time::now();
                gap_pose_pub.publish(gap_array); 
                gap_array.poses.clear();
            } 
            last_pub_P = pub_P;
        }
        
            
            // vector<Point2f> corners_sort_vec(8);
            // for (int i = 0; i < 4; i++)
            // {
            //     corners_sort_vec[i] = out_corners[out_quadr_index_[i]];
            //     corners_sort_vec[i + 4] = in_corners[in_quadr_index_[i]];
            // }
            // float points_ref[8][3] = {{-0.1275f, -0.0845f, 0.0f}, {0.1275f, -0.0845f, 0.0f}, {0.1275f, 0.0845f, 0.0f}, {-0.1275f, 0.0845f, 0.0f},
            //                                         {-0.0975f, -0.0545f, 0.0f}, {0.0975f, -0.0545f, 0.0f}, {0.0975f, 0.0545f, 0.0f}, {-0.0975f, 0.0545f, 0.0f}};
            // vector<Point3f> points_ref_vec(0);
            // for (int i = 0; i < 8; i++)
            // {
            //     points_ref_vec.push_back( Point3f(points_ref[i][0], points_ref[i][1], points_ref[i][2]) );
            // }
            // Mat rvec, tvec;
            // solvePnP(points_ref_vec, corners_sort_vec, cameraMatrix, distCoeffs, rvec, tvec);

            // cv::Matx33f R;
            // Rodrigues(rvec, R);
            // // cout << "plane translation: " << tvec << endl;
            // // cout << "plane norm direction: " << R.col(2) << endl;
            // float centroid_point_ref[3] = {0};
            // for (int i = 0; i < 3; i++)
            // {
            //     centroid_point_ref[i] = tvec.at<float>(i);
            // }
            // float centroid_pixel_ref[2] = {0};
            // rs2_project_point_to_pixel(centroid_pixel_ref, &depth_intrins, centroid_point_ref);
            // vector<Point2f> centroid_pixel_ref_vec(1);
            // centroid_pixel_ref_vec[0] = Point2f(centroid_pixel_ref[0], centroid_pixel_ref[1]);
            // // cout << "centroid piexl from reference: " << centroid_pixel_ref_vec[0] << endl;
            // circle( color_, centroid_pixel_ref_vec[0], radius, Scalar(150, 0, 0), CV_FILLED);
            // rs2_extrinsics plane_extrin;
            // for(int i = 0; i < 3; i++)
            // {
            //     for (int j = 0; j < 3; j++)
            //     {
            //         plane_extrin.rotation[i * 3 + j] = R(j,i);
            //     }
            // }
            // for (int i = 0; i < 3; i++)
            // {
            //     plane_extrin.translation[i] = tvec.at<float>(i);
            // }
            // float points_camera[8][3] = {0};
            // float pixels_camera[8][2] = {0};
            // vector<Point2f> pixels_camera_vec(8);
            // // for (int i = 0; i < 4; i++)
            // // {
            // //     cout << "point from depth: " << points[out_quadr_index[i]][0] << " " << points[out_quadr_index[i]][1] << " " << points[out_quadr_index[i]][2] << endl;
            // // }
            // // for (int i = 0; i < 4; i++)
            // // {
            // //     cout << "point from depth: " << points[in_quadr_index[i]][0] << " " << points[in_quadr_index[i]][1] << " " << points[in_quadr_index[i]][2] << endl;
            // // }
            // for (int i = 0; i < 8; i++)
            // {
            //     rs2_transform_point_to_point(points_camera[i], &plane_extrin, points_ref[i]);
            //     // cout << "point from refer: " << points_camera[i][0] << " " << points_camera[i][1] << " " << points_camera[i][2] << endl;
            //     rs2_project_point_to_pixel(pixels_camera[i], &depth_intrins, points_camera[i]);
            //     pixels_camera_vec[i] = Point2f(pixels_camera[i][0], pixels_camera[i][1]);
            //     circle( color_, pixels_camera_vec[i], radius, Scalar(200, 0, 0), CV_FILLED);
            // }
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
