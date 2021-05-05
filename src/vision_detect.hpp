#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std;

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
