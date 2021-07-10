#include <iostream>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
using namespace cv; 
using namespace std;

// hello realsense 

int maxCorners = 20;
int maxblobColor = 1;
int blobColor = 1;
int minCircularity = 8;
int maxCircularity = 10;
int blobarea = 10;
int maxblobarea = 100;
int minConvexity = 85;
int maxConvexity = 100;
int minInertiaRatio = 30;
int maxInertiaRatio = 100;
int radius = 2;
RNG rng(12345);
const char* source_window = "Source image";
const char* corners_window = "Corners detected";
const char* blob_window = "blob image";
static const std::string OPENCV_WINDOW = "Image window";
void goodFeaturesToTrack_Demo( int, void* );
Mat gray_(Size(848, 480), CV_8UC1);
Mat color(Size(848, 480), CV_8UC3);
Mat color_(Size(848, 480), CV_8UC3);
Mat gray(Size(848, 480), CV_8UC1);
deque<Mat> color_deque;
deque<Mat> gray_deque;
cv_bridge::CvImagePtr cv_ptr;
vector<Point2f> corners_;
float centroid_[3] = {0, 0, 0};
vector<float> direction_ = {0, 0 , 0};
vector<KeyPoint> last_kp;
bool detect_init = false;
bool write_jpg = false;
bool target_update = false;
bool obstacle_update = false;
string jpg_dir = "/home/dji/catkin_ws/src/robocentric_control/image/";

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

bool nearest_pixel_find(KeyPoint &last_kp, vector<KeyPoint> &keypoints, float minsize)
{
    int i = 0;
    int min_pos = 0;
    float distance_min = sqrt(pow(last_kp.pt.x - keypoints.at(0).pt.x, 2) + pow(last_kp.pt.y - keypoints.at(0).pt.y, 2));
    for (i = 1; i < keypoints.size(); i++)
    {   
        float distance = sqrt(pow(last_kp.pt.x - keypoints.at(i).pt.x, 2) + pow(last_kp.pt.y - keypoints.at(i).pt.y, 2));
        if(distance_min > distance)
        {
            distance_min = distance;
            min_pos = i;
        }
    }
    if (keypoints.at(min_pos).size < 2 * minsize && distance_min < 100.0f)
    {
        last_kp = keypoints.at(min_pos);
        keypoints.erase(keypoints.begin() + min_pos);
        return true;
    } 
    else
    {
        cout << "distance: " << distance_min << endl;
        cout << "keypoint size: " << keypoints.at(min_pos).size << endl;
        return false;
    }
}

void blob_detect( int, void* )
{   
    target_update = false;
    obstacle_update = false;
    vector<KeyPoint> keypoints; 
    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;
    // Change thresholds
    params.thresholdStep = 10;
    params.minThreshold = 10;
    params.maxThreshold = 200;
    // Filter by Color.
    params.filterByColor = true;
    params.blobColor = blobColor * 255;
    // Filter by Area.
    params.filterByArea = true;
    params.minArea = blobarea * 10;
    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = (float)minCircularity/10;
    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = (float)minConvexity / 100;
    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = (float)minInertiaRatio / 100;
    // Set up detector with params
    // SimpleBlobDetector detector(params);
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    // You can use the detector this way
    detector->detect( gray_, keypoints);
        // namedWindow(blob_window);
    // std::cout << keypoints.size() << std::endl;
    Mat blob_image;
    namedWindow(blob_window);
        // detector.detect( gray, keypoints);
    drawKeypoints( gray_, keypoints, blob_image, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        // Canny(gray, imgcanny_blur, 100, 200, 3, true);
    if (!detect_init)
    {
        if (keypoints.size() == 2 && sqrt(pow(keypoints.at(0).pt.x - keypoints.at(1).pt.x, 2) + pow(keypoints.at(0).pt.y - keypoints.at(1).pt.y, 2)) < 100.0f)
        {
            if (keypoints.at(0).pt.x < keypoints.at(1).pt.x)
            {
                last_kp.push_back(keypoints.at(0));
                last_kp.push_back(keypoints.at(1));
            }
            else
            {
                last_kp.push_back(keypoints.at(1));
                last_kp.push_back(keypoints.at(0));
            }  
            keypoints.clear();
            detect_init = true; 
            target_update = true;
            for (int i = 0; i < last_kp.size(); i++)
            {
                char num[1];
                sprintf(num, "%d", i);
                cv::putText(blob_image,string(num),last_kp.at(i).pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
            }
            cout << "detect init" << endl;  
        }
        else
        {
            cout << "detect init error" << endl;
        }
    }
    else if (keypoints.size() >= 2)
    {       
        float min_size = min(last_kp.at(0).size, last_kp.at(1).size);
        bool kp_match = true;
        for (int i = 0; i < last_kp.size(); i++)
        {
            kp_match = kp_match && nearest_pixel_find(last_kp.at(i), keypoints, min_size);
        }
        if (kp_match)
        {
            if(last_kp.at(0).pt.x > last_kp.at(1).pt.x)
            {
                swap(last_kp.at(0), last_kp.at(1));
            }
            for (int i = 0; i < last_kp.size(); i++)
            {
                char num[1];
                sprintf(num, "%d", i);
                cv::putText(blob_image,string(num),last_kp.at(i).pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
            }
            target_update = true;
        }
        
    }
    else
    {
        detect_init = false;
        last_kp.clear();
    }

    if(!keypoints.empty())
    {
        float obs_size = keypoints.at(0).size;
        int obs_index = 0;
        for (int i = 0; i < keypoints.size(); i++)
        {
            if(keypoints.at(i).size > obs_size)
            {
                obs_size = keypoints.at(i).size;
                obs_index = i;
            }
        }
        char num[1];
        sprintf(num, "%d", 2);
        cv::putText(blob_image,string(num), keypoints.at(obs_index).pt,cv::FONT_HERSHEY_DUPLEX,0.5,cv::Scalar(0,255,0),2,false);
        obstacle_update = true;
    }
    imshow(blob_window, blob_image);   
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

void color_cb(const sensor_msgs::Image::ConstPtr &msg)
{   
    try
    {
       cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    color = cv_ptr->image;
    char temp[64] ;
    sprintf(temp, "%d", (int)color_deque.size());
    string log_file = jpg_dir + string(temp) + ".jpg";
    imwrite(log_file, color);
    // cvtColor( color, gray, COLOR_BGR2GRAY );
    // blur( gray, gray, Size(3,3) );
    color_deque.push_back(color);
    // if (color_deque.size() < 100)
    // {
    //     color_deque.push_back(color);
    //     gray_deque.push_back(gray);
    // }
    // cout << "c: "<< color_deque.size() << endl;
    // cout << "g: "<< gray_deque.size() << endl;
}

int main(int argc, char* argv[])
{   
    ros::init(argc, argv, "mocap_detect");
    ros::NodeHandle nh;
    
    // rs2::pipeline pipe;
    // rs2::config cfg;
    // // rs2::colorizer color_map;
    // cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 60);
    // cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 60);
    // // cfg.enable_stream(RS2_STREAM_INFRARED, 848, 480, RS2_FORMAT_Y8, 60);
    // pipe.start(cfg);
    // rs2::frameset frames;
    // for(int i = 0; i < 30; i++)
    // {
    //     frames = pipe.wait_for_frames();
    // }
    // rs2::align align_to_color(RS2_STREAM_COLOR);
    // frames = align_to_color.process(frames);
    // rs2::frame color_frame = frames.get_color_frame();
    // rs2::depth_frame depth_frame = frames.get_depth_frame();
    // // rs2::frame ir_frame = frames.first(RS2_STREAM_INFRARED);
    // float width = depth_frame.get_width();
    // float height = depth_frame.get_height();
    // float dist_to_center = depth_frame.get_distance(width/2, height/2);
    // rs2_intrinsics depth_intrins = rs2::video_stream_profile(depth_frame.get_profile()).get_intrinsics();
    // rs2_intrinsics color_intrins = rs2::video_stream_profile(color_frame.get_profile()).get_intrinsics();
    // cv::Matx33f cameraMatrix = {depth_intrins.fx, 0., depth_intrins.ppx, 0, depth_intrins.fy, depth_intrins.ppy, 0, 0 ,0};
    // vector<float> distCoeffs = {depth_intrins.coeffs[0], depth_intrins.coeffs[1], depth_intrins.coeffs[2], depth_intrins.coeffs[3], depth_intrins.coeffs[4]};
    
    // Mat color(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
    // color_ = color;
    // Mat depth(Size(width, height), CV_16UC1, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
    // Mat ir(Size(848, 480), CV_8UC1, (void*)ir_frame.get_data(), Mat::AUTO_STEP);

    ros::Subscriber color_sub = nh.subscribe("/robocentric/camera/color_raw", 1000, color_cb);
    ros::Publisher color_pub = nh.advertise<sensor_msgs::Image>
            ("camera/blob_detect", 10);
    vector<Vec3f> circles;
    ros::Rate rate(10.0);

    // for (int i = 0; i < 10; i++)
    // {
    //     ros::spinOnce();
    // }
    while(ros::ok() && write_jpg)
    {
        ros::spinOnce();
        // imshow(OPENCV_WINDOW, color);
        // blob_detect(0, 0);
        // imshow(OPENCV_WINDOW, gray_);
        // waitKey();
        std_msgs::Header header;
        cv_bridge::CvImage img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, color_);
        sensor_msgs::Image img_msg;
        img_bridge.toImageMsg(img_msg);
        color_pub.publish(img_msg);
        rate.sleep();
    }
    int i = 0;
    double total_time = 0.0;
    while(1)
    {
        char temp[64] ;
        sprintf(temp, "%d", i);
        string jpg_name = jpg_dir + string(temp) + ".jpg";
        color = imread(jpg_name, IMREAD_COLOR);
        cvtColor( color, gray_, COLOR_BGR2GRAY );
        blur( gray_, gray_, Size(3,3) );
        // namedWindow( blob_window );
        double time1 = ros::Time::now().toSec();
        blob_detect(0, 0);
        double time2 = ros::Time::now().toSec();
        total_time += time2 - time1;
        if (i % 100 == 0)
        {
            cout << i << endl;
            cout << "blob_average time: " << total_time/(i+1) << endl;
        }

        // createTrackbar( "blob color:", blob_window, &blobColor, maxblobColor, blob_detect );
        // createTrackbar( "blob minCircularity:", blob_window, &minCircularity, maxCircularity, blob_detect );
        // createTrackbar( "blob minarea:", blob_window, &blobarea, maxblobarea, blob_detect );
        // createTrackbar( "blob minconvexity:", blob_window, &minConvexity, maxConvexity, blob_detect );
        // createTrackbar( "blob minInertiaRatio:", blob_window, &minInertiaRatio, maxInertiaRatio, blob_detect );
        // imshow(blob_window, color); 
        // blob_detect(0, 0);
        char key = (char)waitKey();
        if (key == '0')
        {
            break;
        }
        else if (key == '2')
        {   
            i--;
        }
        else if(key == '1')
        {   
            i++;
        }
        // i++;    
    }
    
    return 0;
}
