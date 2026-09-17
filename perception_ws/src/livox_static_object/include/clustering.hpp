#pragma once
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <Eigen/Dense>

namespace clustering {

inline double autoDistance(const pcl::PointCloud<pcl::PointXYZI>& cloud, double base)
{
    // 평균 인접 거리 → base 배
    if (cloud.empty()) return base;
    double sum = 0.0;
    for (size_t i = 1; i < cloud.size(); ++i)
        sum += (cloud.points[i].getVector3fMap() -
                cloud.points[i-1].getVector3fMap()).norm();
    double avg = sum / static_cast<double>(cloud.size());
    return std::max(base, 1.5 * avg);
}

inline std::vector<pcl::PointCloud<pcl::PointXYZI>> euclidean(
        const pcl::PointCloud<pcl::PointXYZI>& cloud,
        double cluster_dist,
        int min_pts)
{
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud.makeShared());

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_dist);
    ec.setMinClusterSize(min_pts);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud.makeShared());
    ec.extract(cluster_indices);

    std::vector<pcl::PointCloud<pcl::PointXYZI>> clusters;
    for (auto& idx : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZI> c;
        for (int id : idx.indices) c.points.push_back(cloud.points[id]);
        clusters.push_back(std::move(c));
    }
    return clusters;
}

inline Eigen::Vector3f centroid(const pcl::PointCloud<pcl::PointXYZI>& c)
{
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (auto& p : c.points) sum += p.getVector3fMap();
    return sum / static_cast<float>(c.size());
}

inline Eigen::Vector3f eigenValues(const pcl::PointCloud<pcl::PointXYZI>& c)
{
    Eigen::MatrixXf mat(3, c.size());
    for (size_t i = 0; i < c.size(); ++i) mat.col(i) = c.points[i].getVector3fMap();
    Eigen::Vector3f mean = mat.rowwise().mean();
    Eigen::MatrixXf demean = mat.colwise() - mean;
    Eigen::Matrix3f cov = demean * demean.transpose() / static_cast<float>(c.size());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
    Eigen::Vector3f eval = es.eigenvalues();
    return Eigen::Vector3f(eval(2), eval(1), eval(0)).reverse(); // λ₁≥λ₂≥λ₃
}

} // namespace
