#pragma once

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include "params.h"
#include "clustering.hpp"
#include "shape_filter.hpp"
#include "intensity_filter.hpp"
#include "height_filter.hpp"

namespace livox_static_object
{
class StaticObjectDetector
{
public:
    StaticObjectDetector(ros::NodeHandle& nh, ros::NodeHandle& pnh);
private:
    void cloudCB(const sensor_msgs::PointCloud2ConstPtr& msg);

    ros::Subscriber sub_;
    ros::Publisher  pub_;

    std::string input_topic_;
    std::string output_topic_;
    double utm_x_{};
    double utm_y_{};
};
} // namespace
