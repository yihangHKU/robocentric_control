#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

using namespace cv; 
using namespace std;
std::ofstream fout_vision;
std::ofstream fout_depth;
int verbose = 1;
vector<Eigen::Matrix<float, 3, 1>> P_buffer;
vector<Eigen::Matrix<float, 3, 1>> P2_buffer;
vector<Eigen::Quaternionf> q_buffer;

float points_distance(float points1[3], float points2[3])
{
    return sqrt(pow(points1[0] - points2[0], 2) + pow(points1[1] - points2[1], 2) + pow(points1[2] - points2[2], 2));
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

Eigen::Matrix<float, 3, 1> ENU_2_NED(Eigen::Matrix<float, 3, 1> vector)
{
    float buffer = vector[0];
    vector[0] = vector[1];
    vector[1] = buffer;
    vector[2] = -vector[2];
    return vector;
}

Eigen::Matrix<float, 3, 1> FLU_2_FRD(Eigen::Matrix<float, 3, 1> vector)
{
	vector[1] = - vector[1];
	vector[2] = - vector[2];
	return vector;
}

vector<float> vector_normalize(vector<float> vector)
{   
    float length = sqrt(pow(vector[0], 2) + pow(vector[1], 2) + pow(vector[2], 2));
    vector[0] = vector[0] / length;
    vector[1] = vector[1] / length;
    vector[2] = vector[2] / length;
    return vector;
}

float parallel_check(float vector1[3], float vector2[3])
{
    return abs(vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector1[2]);
}

void plane_from_points(vector<Point3f> points, vector<float> &direction_, vector<float> &centroid_)
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

float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
    uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

    int width = other_frame.get_width();
    int height = other_frame.get_height();
    int other_bpp = other_frame.get_bytes_per_pixel();

    #pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
    for (int y = 0; y < height; y++)
    {
        auto depth_pixel_index = y * width;
        for (int x = 0; x < width; x++, ++depth_pixel_index)
        {
            // Get the depth value of the current pixel
            auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

            // Check if the depth value is invalid (<=0) or greater than the threashold
            // if (pixels_distance <= 0.f || pixels_distance > clipping_dist)
            if (pixels_distance > clipping_dist)
            {
                // Calculate the offset in other frame's buffer to current pixel
                auto offset = depth_pixel_index * other_bpp;

                // Set pixel to "background" color (0x999999)
                std::memset(&p_other_frame[offset], 0, other_bpp);
            }
        }
    }
}
bool blob_detect_init(Ptr<SimpleBlobDetector> &detector)
{
    int blobColor = 255;
    float minCircularity = 0.8;
    int blobarea = 60;
    float minConvexity = 0.72;
    float minInertiaRatio = 0.8;
    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;
    // Change thresholds
    params.thresholdStep = 10;
    params.minThreshold = 10;
    params.maxThreshold = 200;
    // Filter by Color.
    params.filterByColor = true;
    params.blobColor = blobColor ;
    // Filter by Area.
    params.filterByArea = true;
    params.minArea = blobarea ;
    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = minCircularity;
    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = minConvexity;
    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = minInertiaRatio;
    // Set up detector with params
    // SimpleBlobDetector detector(params);
    detector = SimpleBlobDetector::create(params);
    return true;
}

void blob_detect(Ptr<SimpleBlobDetector> &detector, Mat &gray, Mat &src, rs2::depth_frame &depth_frame)
{   
    vector<KeyPoint> keypoints;
    rs2_intrinsics depth_intrins = rs2::video_stream_profile(depth_frame.get_profile()).get_intrinsics();
    // You can use the detector this way
    double time2 = ros::Time::now().toSec();
    detector->detect( gray, keypoints);
    double time3 = ros::Time::now().toSec();
    // namedWindow(blob_window);
    // std::cout << keypoints.size() << std::endl; 
    // Mat blob_image;
    // detector.detect( gray, keypoints);
    if (!keypoints.empty())
    {
        drawKeypoints( src, keypoints, src, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        for (int i = 0; i < keypoints.size(); i++)
        {   
            float points[3];
            float pixels[2] = {keypoints[i].pt.x, keypoints[i].pt.y}; 
            float depth = depth_frame.get_distance(keypoints[i].pt.x, keypoints[i].pt.y);
            rs2_deproject_pixel_to_point(points, &depth_intrins, pixels, depth);
            vector<float> circle_center = {points[0], points[1], points[2]};
            circle_center = RDF_2_FRD(circle_center);
            std::cout << "circle center: " << circle_center[0] << " " << circle_center[1] << " " << circle_center[2] << std::endl;
            Eigen::Matrix<float, 3, 1> P;
            P << circle_center[0], circle_center[1], circle_center[2];
            P_buffer.push_back(P);
        }
    }
    // std::cout << "4-1:" << time4-time1<<std::endl;
    std::cout << "3-2:" << time3-time2<<std::endl;
    // imshow(blob_window, blob_image);
}

bool gap_detect(Mat &gray, rs2::depth_frame &depth_frame, Mat &color_)
{   
    int radius = 4;
    RNG rng(12345);
    int step = 0;
    vector<Point2f> corners_;
    int width = depth_frame.get_width();
    int height = depth_frame.get_height();
    rs2_intrinsics depth_intrins = rs2::video_stream_profile(depth_frame.get_profile()).get_intrinsics();
    cv::Matx33f cameraMatrix = {depth_intrins.fx, 0., depth_intrins.ppx, 0, depth_intrins.fy, depth_intrins.ppy, 0, 0 ,0};
    vector<float> distCoeffs = {depth_intrins.coeffs[0], depth_intrins.coeffs[1], depth_intrins.coeffs[2], depth_intrins.coeffs[3], depth_intrins.coeffs[4]};
    Mat imgcanny_blur;
    Canny(gray, imgcanny_blur, 100, 200, 3, true);
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
        vector<float> direction_ = {0, 0, 0};
        vector<float> centroid = {0, 0, 0};
        plane_from_points(points_vec, direction_, centroid);
        float centroid_[3] = {centroid[0], centroid[1], centroid[2]};
        // cout << "publish time: " << ros::Time::now() << endl;
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
        step++;
        return depth_error;
    }
}