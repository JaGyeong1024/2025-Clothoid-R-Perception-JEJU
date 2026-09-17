#include "livox_static_object.h"
#include <Eigen/Dense>

using namespace livox_static_object;

StaticObjectDetector::StaticObjectDetector(ros::NodeHandle& nh, ros::NodeHandle& pnh)
{
    pnh.param<std::string>("input_topic",  input_topic_,  "/livox/lidar");
    pnh.param<std::string>("output_topic", output_topic_, "/static_obstacles");
    pnh.param<double>("utm_offset_x", utm_x_, 0.0);
    pnh.param<double>("utm_offset_y", utm_y_, 0.0);

    sub_ = nh.subscribe(input_topic_, 1, &StaticObjectDetector::cloudCB, this);
    pub_ = nh.advertise<sensor_msgs::PointCloud>(output_topic_, 1);
}

void StaticObjectDetector::cloudCB(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud);

    // 0. 지면 & 높이 필터
    auto roi_cloud = height_filter::passZ(*cloud, ls_param::Z_MIN, ls_param::Z_MAX);

    // 1. Euclidean clustering (auto-scaled)
    double dist_thr = clustering::autoDistance(*roi_cloud, ls_param::CLUSTER_DIST);
    auto clusters = clustering::euclidean(*roi_cloud, dist_thr, ls_param::MIN_CLUSTER_PTS);

    std::vector<Eigen::Vector3f> centroids;
    for (auto& c : clusters)
    {
        if (shape_filter::isVerticalSlender(c, ls_param::VERTICAL_RATIO))    continue; // pole/tree
        if (!shape_filter::isCylinderOrCone(c, ls_param::CYL_DIST_THRESH,
                                               ls_param::CONE_DIST_THRESH,
                                               ls_param::INLIER_RATIO))     continue;
        if (!intensity_filter::inRange(c, ls_param::INTENSITY_LOW, ls_param::INTENSITY_HIGH)) continue;
        if (!height_filter::zStdBelow(c, ls_param::Z_STD_MAX))               continue;
        if (shape_filter::isLongCurb(c, ls_param::CURB_LEN_RATIO))           continue;

        centroids.emplace_back(clustering::centroid(c));
    }

    sensor_msgs::PointCloud out;
    out.header = msg->header;
    out.header.frame_id = "map";
    out.points.reserve(centroids.size());

    for (auto& v : centroids)
    {
        geometry_msgs::Point32 p;
        p.x = v.x() + utm_x_;
        p.y = v.y() + utm_y_;
        p.z = v.z();
        out.points.push_back(p);
    }
    pub_.publish(out);
}
