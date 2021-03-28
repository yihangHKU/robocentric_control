#include <ros/ros.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace cv; 
using namespace std;

// hello realsense 

int maxCorners = 20;
int maxTrackbar = 100;
int radius = 2;
RNG rng(12345);
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
Mat color_, gray_;
vector<Point2f> corners_;
float centroid_[3] = {0, 0, 0};
vector<float> direction_ = {0, 0, 0};

sensor_msgs::Image color_frame;
sensor_msgs::Image depth_frame;
sensor_msgs::CameraInfo color_info;
geometry_msgs::PoseStamped gap_pose;

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
    cout << "plane centroid: ";
    for (int i = 0; i < 3; i++)
    {
        cout << " " << centroid_[i];
    }
    cout << endl;
    cout << "plane norm direction: ";
    for (int i = 0; i < 3; i++)
    {
        cout << " " << direction_[i];
    }
    cout << endl;
}

float points_distance(float points1[3], float points2[3])
{
    return sqrt(pow(points1[0] - points2[0], 2) + pow(points1[1] - points2[1], 2) + pow(points1[2] - points2[2], 2));
}

vector<float> vector_normalize(vector<float> vector)
{   
    float length = sqrt(pow(vector[0], 2) + pow(vector[1], 2) + pow(vector[2], 2));
    vector[0] = vector[0] / length;
    vector[1] = vector[1] / length;
    vector[2] = vector[2] / length;
    return vector;
}

vector<float> vector_product(vector<float> vector1, vector<float> vector2)
{   
    vector<float> vector3 = {0., 0., 0.};
    vector3[0] = vector1[1] * vector2[2] - vector1[2] * vector2[1];
    vector3[1] = vector1[2] * vector2[0] - vector1[0] * vector2[2];
    vector3[2] = vector1[0] * vector2[1] - vector1[1] * vector2[0];
    return vector3;
}

vector<float> RDF_2_FRD(vector<float> vector)
{
    float buffer = vector[0];
    vector[0] = vector[2];
    vector[2] = vector[1];
    vector[1] = buffer;
    return vector;
}

float parallel_check(float vector1[3], float vector2[3])
{
    return abs(vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector1[2]);
}

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "state_estim");
    ros::NodeHandle nh;

    ros::Publisher gap_pose_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("camera/gap_pose", 100);

    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::colorizer color_map;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 60);
    // cfg.enable_stream(RS2_STREAM_INFRARED, 1280, 720, RS2_FORMAT_Y8, 30);
    pipe.start(cfg);
    rs2::frameset frames;
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    ros::Rate rate(100.0);
    for(int i = 0; i < 10; i++)
    {
        frames = pipe.wait_for_frames();
        // ros::spinOnce();
    }
    while(ros::ok()){ 
        frames = pipe.wait_for_frames();
        rs2::align align_to_color(RS2_STREAM_COLOR);
        frames = align_to_color.process(frames);
        rs2::frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();
        // rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
        int width = depth_frame.get_width();
        int height = depth_frame.get_height();
        float dist_to_center = depth_frame.get_distance(width/2, height/2);
        // float dist_to_center = depth_frame.data[int(width/2)][int(height/2)];
        rs2_intrinsics depth_intrins = rs2::video_stream_profile(depth_frame.get_profile()).get_intrinsics();
        rs2_intrinsics color_intrins = rs2::video_stream_profile(color_frame.get_profile()).get_intrinsics();
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
        blur( gray, gray, Size(3,3) );
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
                cout << "gap index: " << *it << endl;
            }
        }
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
                circle( color_, out_corners[i], radius, Scalar(0, 0, 200), CV_FILLED );
                circle( color_, in_corners[i], radius, Scalar(0, 0, 200), CV_FILLED );
            }
            float out_points[4][3];
            float out_pixels[4][2];
            float in_points[4][3];
            float in_pixels[4][2];
            vector<Point3f> points_vec(0);
            vector<Point2f> pixels_vec(0);
            for (int i = 0; i < 4; i++)
            {   
                // cout << "corner pixels depth: " << depth_frame.get_distance(out_corners[i].x, out_corners[i].y) << endl;
                out_pixels[i][0] = out_corners[i].x;
                out_pixels[i][1] = out_corners[i].y;
                in_pixels[i][0] = in_corners[i].x;
                in_pixels[i][1] = in_corners[i].y;
                rs2_deproject_pixel_to_point(out_points[i], &depth_intrins, out_pixels[i], depth_frame.get_distance(out_corners[i].x, out_corners[i].y));
                rs2_deproject_pixel_to_point(in_points[i], &depth_intrins, in_pixels[i], depth_frame.get_distance(in_corners[i].x, in_corners[i].y));
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
            circle( color_, centroid_pixel_[0], radius, Scalar(0, 0, 200), CV_FILLED );
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
            cout << "plane x direction: ";
            for (int i = 0; i < 3; i++)
            {
               cout << " " << gap_direction_x[i];
            }
            cout << endl;
            cout << "plane y direction: ";
            for (int i = 0; i < 3; i++)
            {
               cout << " " << gap_direction_y[i];
            }
            cout << endl;
            vector<float> gap_center = {centroid_[0], centroid_[1], centroid_[2]};
            gap_center = RDF_2_FRD(gap_center);
            gap_direction_x = RDF_2_FRD(gap_direction_x);
            gap_direction_y = RDF_2_FRD(gap_direction_y);
            gap_direction_z = RDF_2_FRD(gap_direction_z);
            Eigen::Matrix<float, 3, 3> R;
            R << gap_direction_z[0], gap_direction_z[1], gap_direction_z[2],
                 gap_direction_x[0], gap_direction_x[1], gap_direction_x[2],
                 gap_direction_y[0], gap_direction_y[1], gap_direction_y[2];
            Eigen::Matrix<float, 3, 1> P;
            P << gap_center[0], gap_center[1], gap_center[2];
            Eigen::Quaternionf q(R);

            gap_pose.pose.position.x = gap_center[0];
            gap_pose.pose.position.y = gap_center[1];
            gap_pose.pose.position.z = gap_center[2];
            gap_pose.pose.orientation.x = q.x();
            gap_pose.pose.orientation.y = q.y();
            gap_pose.pose.orientation.z = q.z();
            gap_pose.pose.orientation.w = q.w();
            gap_pose.header.stamp = ros::Time::now();
            gap_pose_pub.publish(gap_pose);
            
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
        }
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
