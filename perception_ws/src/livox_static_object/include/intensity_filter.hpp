#pragma once
#include <pcl/point_cloud.h>

namespace intensity_filter {
inline bool inRange(const pcl::PointCloud<pcl::PointXYZI>& c, double low, double high)
{
    double sum = 0.0;
    for (auto& p : c.points) sum += p.intensity;
    double mean = sum / c.size();
    return (mean >= low && mean <= high);
}
} // namespace
