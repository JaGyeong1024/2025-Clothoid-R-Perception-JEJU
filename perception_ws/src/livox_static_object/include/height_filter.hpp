#pragma once
#include <pcl/filters/passthrough.h>

namespace height_filter {

inline pcl::PointCloud<pcl::PointXYZI>::Ptr passZ(
        const pcl::PointCloud<pcl::PointXYZI>& cloud,
        double zmin, double zmax)
{
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud.makeShared());
    pass.setFilterFieldName("z");
    pass.setFilterLimits(zmin, zmax);
    pcl::PointCloud<pcl::PointXYZI>::Ptr out(new pcl::PointCloud<pcl::PointXYZI>);
    pass.filter(*out);
    return out;
}

inline bool zStdBelow(const pcl::PointCloud<pcl::PointXYZI>& c, double std_max)
{
    if (c.empty()) return false;
    double mean = 0.0;
    for (auto& p : c.points) mean += p.z;
    mean /= c.size();
    double var = 0.0;
    for (auto& p : c.points) var += (p.z - mean) * (p.z - mean);
    var /= c.size();
    return std::sqrt(var) <= std_max;
}

} // namespace
