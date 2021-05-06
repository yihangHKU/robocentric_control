#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
using namespace cv; 
using namespace std;

// hello realsense 

int maxCorners = 20;
int maxTrackbar = 100;
int radius = 2;
RNG rng(12345);
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
void goodFeaturesToTrack_Demo( int, void* );
Mat color_, gray_;
vector<Point2f> corners_;
float centroid_[3] = {0, 0, 0};
vector<float> direction_ = {0, 0 , 0};

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

void goodFeaturesToTrack_Demo( int, void* )
{
    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    Mat copy = color_.clone();
    goodFeaturesToTrack( gray_,
                        corners,
                        maxCorners,
                        qualityLevel,
                        minDistance,
                        Mat(),
                        blockSize,
                        gradientSize,
                        useHarrisDetector,
                        k );
    cout << "** Number of corners detected: " << corners.size() << endl;
    corners_ = corners;
    int radius = 4;
    for( size_t i = 0; i < corners.size(); i++ )
    {
        // circle( color_, corners[i], radius, Scalar(rng.uniform(0,255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED );
        circle( color_, corners[i], radius, Scalar(0, 0, 200), FILLED );
    }
    namedWindow( source_window );
    imshow( source_window, color_);
}

float points_distance(float points1[3], float points2[3])
{
    return sqrt(pow(points1[0] - points2[0], 2) + pow(points1[1] - points2[1], 2) + pow(points1[2] - points2[2], 2));
}

float* vector_normalize(float vector[3])
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


int main()
{  
    rs2::pipeline pipe;
    rs2::config cfg;
    // rs2::colorizer color_map;
    cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 60);
    // cfg.enable_stream(RS2_STREAM_INFRARED, 848, 480, RS2_FORMAT_Y8, 60);
    pipe.start(cfg);
    rs2::frameset frames;
    for(int i = 0; i < 30; i++)
    {
        frames = pipe.wait_for_frames();
    }
    rs2::align align_to_color(RS2_STREAM_COLOR);
    // rs2::align align_to_depth(RS2_STREAM_DEPTH);
    frames = align_to_color.process(frames);
    rs2::frame color_frame = frames.get_color_frame();
    rs2::depth_frame depth_frame = frames.get_depth_frame();
    // rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
    float width = depth_frame.get_width();
    float height = depth_frame.get_height();
    float dist_to_center = depth_frame.get_distance(width/2, height/2);
    rs2_intrinsics depth_intrins = rs2::video_stream_profile(depth_frame.get_profile()).get_intrinsics();
    rs2_intrinsics color_intrins = rs2::video_stream_profile(color_frame.get_profile()).get_intrinsics();
    cv::Matx33f cameraMatrix = {depth_intrins.fx, 0., depth_intrins.ppx, 0, depth_intrins.fy, depth_intrins.ppy, 0, 0 ,0};
    vector<float> distCoeffs = {depth_intrins.coeffs[0], depth_intrins.coeffs[1], depth_intrins.coeffs[2], depth_intrins.coeffs[3], depth_intrins.coeffs[4]};
    
    // for (int i = 0; i<5; i++)
    // {    
    //     cout << "depth distcoeffs: " << distCoeffs[i] << endl;
    // }
    // cout << "depth intrinsics ppx" << cameraMatrix[0][2] << endl;
    // rs2::frame colorized_depth = color_map.colorize(depth_frame);
    Mat color(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
    color_ = color;
    Mat depth(Size(width, height), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
    // Mat ir(Size(848, 480), CV_8UC1, (void*)ir_frame.get_data(), Mat::AUTO_STEP); 
    Mat gray;
    // Mat imgcanny;
    Mat imgcanny_blur;
    cvtColor( color, gray, COLOR_BGR2GRAY );
    gray_ = gray;
    vector<Vec3f> circles;
    HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows/4, 50, 10);
    cout << "circle size: " << circles.size() << endl;
    for (size_t i = 0; i < circles.size(); i++)
    {
        Point2f center(circles[i][0],circles[i][1]);
        float radius = circles[i][2];
        circle(imgcanny_blur, center, radius, Scalar(255, 0, 0), 3, 8, 0);
    }
    blur( gray, gray, Size(3,3) );
    Canny(gray, imgcanny_blur, 100, 200, 3, true);
    namedWindow("imgCanny blur", CV_WINDOW_AUTOSIZE);
    imshow("imgCanny blur", imgcanny_blur);
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
            double length_contour = arcLength(contours[i], false);
            // Scalar color = Scalar(100, 100, 100);
            // drawContours( imgcontour, contours, (int)i, color, 1, LINE_8, hierarchy, 0);
            cout << "contour index: " << i << " length: " << length_contour << "hierarchy: " << hierarchy[i] << endl;
        }
        if (contours[i].size() >= 10)
        {   
            Scalar color = Scalar(100, 100, 100);
            std::cout << "contours size: "<< contours[i].size() <<std::endl;
            drawContours(imgcontour, contours, i, color, 1, LINE_8, hierarchy, 0);
            Point2f center;
            float radius;
            minEnclosingCircle(contours[i], center, radius);
            cout<< "center: " << center.x << " "<<center.y << " radius:" << radius << endl;
            circle(imgcontour, center, cvRound(radius), Scalar(0,255,0), 1, LINE_AA);
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
        drawContours( imgcontour, contours, (int)*it, color, 1, LINE_8, hierarchy, 0);
        int inner_contour_index = hierarchy[hierarchy[hierarchy[*it][2]][2]][2];
        drawContours( imgcontour, contours, (int)inner_contour_index, color, 1, LINE_8, hierarchy, 0);
        cout << "outer corners: " << contours[*it] << endl;
        cout << "inner corners: " << contours[inner_contour_index] << endl;
        vector<Point2f> out_corners;
        vector<Point2f> in_corners;
        for (int i = 0; i < 4; i ++)
        {
            out_corners.push_back(contours[*it][i]);
            in_corners.push_back(contours[inner_contour_index][i]);
            circle( color_, out_corners[i], radius, Scalar(0, 0, 200), FILLED );
            circle( color_, in_corners[i], radius, Scalar(0, 0, 200), FILLED );
        }
        float out_points[4][3];
        float out_pixels[4][2];
        float in_points[4][3];
        float in_pixels[4][2];
        vector<Point3f> points_vec(0);
        vector<Point2f> pixels_vec(0);
        int range = 5;
        for (int i = 0; i < 4; i++)
        {   
            cout << "origin out corner pixels depth: " << depth_frame.get_distance(out_corners[i].x, out_corners[i].y) << endl;
            cout << "origin in corner pixels depth: " << depth_frame.get_distance(in_corners[i].x, in_corners[i].y) << endl;
            float out_corner_depth = depth_frame.get_distance(out_corners[i].x, out_corners[i].y);
            float in_corner_depth = depth_frame.get_distance(in_corners[i].x, in_corners[i].y);
            for (int j = -range; j < range+1; j++)
            {
                for(int k = -range; k < range+1; k++)
                {
                    float depth_search = depth_frame.get_distance(out_corners[i].x + k, out_corners[i].y + j);
                    if(depth_search > 0.20 && (depth_search < out_corner_depth || out_corner_depth < 0.20))
                    {
                        out_corner_depth = depth_search;
                    }
                    depth_search = depth_frame.get_distance(in_corners[i].x + k, in_corners[i].y + j);
                    if(depth_search > 0.20 && (depth_search < in_corner_depth || in_corner_depth < 0.20))
                    {
                        in_corner_depth = depth_search;
                    }
                }
            }
            cout << "searched out corner pixels depth: " << out_corner_depth << endl;
            cout << "searched in corner pixels depth: " << in_corner_depth << endl;
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
        float centroid_pixel[2] = {0, 0};
        rs2_project_point_to_pixel(centroid_pixel, &depth_intrins, centroid_);
        cout << "centroid pixel: " << centroid_pixel[0] << " " << centroid_pixel[1] << endl;
        vector<Point2f> centroid_pixel_(1);
        centroid_pixel_[0] = Point2f(centroid_pixel[0], centroid_pixel[1]);
        circle( color_, centroid_pixel_[0], radius, Scalar(0, 0, 200), FILLED );
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
        vector<Point2f> corners_sort_vec(8);
        for (int i = 0; i < 4; i++)
        {
            corners_sort_vec[i] = out_corners[out_quadr_index_[i]];
            corners_sort_vec[i + 4] = in_corners[in_quadr_index_[i]];
        }
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
        // cout << "plane translation: " << tvec << endl;
        // cout << "plane norm direction: " << R.col(2) << endl;
        // float centroid_point_ref[3] = {0};
        // for (int i = 0; i < 3; i++)
        // {
        //     centroid_point_ref[i] = tvec.at<float>(i);
        // }
        // float centroid_pixel_ref[2] = {0};
        // rs2_project_point_to_pixel(centroid_pixel_ref, &depth_intrins, centroid_point_ref);
        // vector<Point2f> centroid_pixel_ref_vec(1);
        // centroid_pixel_ref_vec[0] = Point2f(centroid_pixel_ref[0], centroid_pixel_ref[1]);
        // cout << "centroid piexl from reference: " << centroid_pixel_ref_vec[0] << endl;
        // circle( color_, centroid_pixel_ref_vec[0], radius, Scalar(150, 0, 0), FILLED);
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
        //     circle( color_, pixels_camera_vec[i], radius, Scalar(200, 0, 0), FILLED);
        // }
    }
    namedWindow("imgContour", CV_WINDOW_AUTOSIZE);
    imshow("imgContour", imgcontour);
    namedWindow( source_window );
    imshow( source_window, color_);
    // cv::cvtColor(color, color, COLOR_RGB2BGR);

    // Apply Histogram Equalization
    Mat depth_new;
    Mat depth_nnew;
    // equalizeHist( depth, depth_new);
    depth.convertTo(depth_new, CV_8UC1, 0.1, 0); 
    // applyColorMap(depth_new, depth_nnew, COLORMAP_JET);
    // depth.convertTo(depth_new, CV_8UC1, 1, 0); 
    // namedWindow("DISPLAY IMAGE", WINDOW_AUTOSIZE);
    namedWindow("DISPLAY IMAGE NEW", WINDOW_AUTOSIZE);
    // imshow("DISPLAY IMAGE", depth_nnew);
    imshow("DISPLAY IMAGE NEW", depth_new);
    // namedWindow( source_window );
    // createTrackbar( "Max corners:", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo );
    // imshow( source_window, color );

    // goodFeaturesToTrack_Demo( 0, 0 );
    // // Mat rvec, tvec;
    // // solvePnP(points_vec, pixels_vec, cameraMatrix, distCoeffs, rvec, tvec);
    // // cv::Matx33f R;
    // // Rodrigues(rvec, R);
    // float distance[8] = {0};
    // float distance_sort[8] = {0};
    // for (int i = 0; i < corners_.size(); i++)
    // {
    //     distance[i] = sqrt(pow(points_vec[i].x - centroid_[0], 2) +  pow(points_vec[i].y - centroid_[1], 2) + pow(points_vec[i].z - centroid_[2], 2));
    //     distance_sort[i] = distance[i];
    //     cout << "distance from center: " << distance[i] << endl;
    // }
    // sort(distance_sort, distance_sort + 8);
    // int in_quadr_index[4] = {0};
    // int out_quadr_index[4] = {0};
    // for (int i = 0; i < 4; i++) // find in_quadr_index
    // {   
    //     int j = 0;
    //     while (1)
    //     {
    //         if (abs(distance_sort[i] - distance[j]) < 1e-8f)
    //         {
    //             in_quadr_index[i] = j;
    //             cout << "in index: " << j << endl;
    //             break;
    //         }
    //         j++;   
    //     }
    // }
    // for (int i = 0; i < 4; i++) // find out_quadr_index
    // {
    //     int j = 0;
    //     while(1)
    //     {
    //         if (abs(distance_sort[7-i] - distance[j]) < 1e-8f)
    //         {
    //             out_quadr_index[i] = j;
    //             cout << "out index: " << j << endl;
    //             break;
    //         }
    //         j++;
    //     }
    // } 
    waitKey(0);
    return 0;
}
