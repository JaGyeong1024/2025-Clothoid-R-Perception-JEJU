// detection.h
#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/PointCloud2.h>
#include <detect_msgs/Yolo_Objects.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vector>
#include <map>

/* ===== Parameters ===== */
static constexpr double BBOX_SCALE_RATIO = 0.8;
static constexpr double GROUND_THRESH = 0.0;
static constexpr double CLUSTER_TOLERANCE = 0.4;
static constexpr int CLUSTER_MIN_SIZE = 0;
static constexpr int CLUSTER_MAX_SIZE = 100;
static constexpr double ROI_RADIUS_PX = 10.0;
static constexpr double MATCH_DIST = 7.0;
static constexpr int TRACKER_MAX_MISS = 15;
static constexpr int MIN_BBOX_EDGE_PX = 0;

/* ===== Kalman Tracker ===== */
struct KalmanTracker
{
    int id{-1};
    int miss_count{0};
    cv::KalmanFilter kf;
    cv::Point2f last_pos;

    KalmanTracker() = default;
    KalmanTracker(const cv::Point2f &pt, int tracker_id, float dt = 0.1f);
    cv::Point2f predict();
    void update(const cv::Point2f &pt);
    void miss();
};

/* ===== Object_Detection ===== */
class Object_Detection
{
private:
    ros::NodeHandle nh;
    ros::Publisher cloud_minjae_pub;  // publishes centroids on "/minjae"
    ros::Publisher cloud_fillter_pub; // publishes filtered 3D points on "/cloud_fillter"

    std::string lidar_topic, camera_topic, yolo_topic, frame_name;

    cv::Mat projection_matrix;
    cv::Mat camera_image;
    std::vector<cv::Point3d> lidar_points;
    std::vector<cv::Point2d> projected_list;
    std::vector<cv::Point2d> prev_centroids;

    void read_projection_matrix();
    void detectionCallback(const sensor_msgs::PointCloud2::ConstPtr &lidar_msg,
                           const sensor_msgs::CompressedImage::ConstPtr &camera_msg,
                           const detect_msgs::Yolo_Objects::ConstPtr &yolo_msg);
    void convert_msg(const detect_msgs::Yolo_Objects::ConstPtr &yolo_msg,
                     const std_msgs::Header &header);
    void publish_2D_pointcloud(const std::vector<cv::Point2d> &pts,
                               const std_msgs::Header &header);
    void track_and_visualize(const std::vector<cv::Point2d> &cents);
    void match_and_update_trackers(const std::vector<cv::Point2f> &cents,
                                   double match_dist = MATCH_DIST,
                                   int max_miss = TRACKER_MAX_MISS);
    std::vector<int> remove_ground_ransac(const std::vector<cv::Point3f> &pts,
                                          double threshold = GROUND_THRESH);

public:
    explicit Object_Detection(ros::NodeHandle *nh);
    ~Object_Detection();
};