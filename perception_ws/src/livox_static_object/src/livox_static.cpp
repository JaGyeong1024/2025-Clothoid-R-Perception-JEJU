// src/livox_static.cpp
#include <array>
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

// PCL GPU modules
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/octree/octree.h>
#include <pcl/segmentation/extract_clusters.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using GPUCloud = pcl::gpu::DeviceArray<PointT>;

// --- 파라미터 설정 (상단 고정) ---
struct Params {
    float roi_x_min = -10.0f, roi_x_max = 10.0f;
    float roi_y_min = -5.0f,  roi_y_max = 5.0f;
    float roi_z_min = -5.0f,  roi_z_max = 5.0f;
    float voxel_size     = 0.1f;
    float ground_thresh  = 0.2f;
    float tilt_angle_deg = 3.3f;
    float base_tol       = 0.5f;
    float alpha_tol      = 0.05f;
    int   min_cluster_size = 5;
    int   max_cluster_size = 25000;
    std::array<float,2> cone_h = {0.5f, 1.0f};
    std::array<float,2> cone_d = {0.2f, 0.5f};
    std::array<float,2> drum_h = {0.8f, 1.3f};
    std::array<float,2> drum_d = {0.4f, 1.0f};
} p;

// Publishers
static ros::Publisher clustered_pub;
static ros::Publisher static_pub;

// GPU filter objects (재사용)
static pcl::gpu::CropBox                    gpu_cropbox;
static pcl::gpu::VoxelGrid                 gpu_voxel;
static pcl::gpu::StatisticalOutlierRemoval gpu_sor;
static pcl::gpu::Octree                    gpu_octree;
static pcl::gpu::EuclideanClusterExtraction gpu_ec;

void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    // 시간 측정 시작
    ros::Time start = ros::Time::now();

    // 1) ROS → PCL
    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) return;

    // 2) 틸트 보정
    float tilt_rad = p.tilt_angle_deg * M_PI / 180.0f;
    Eigen::Affine3f tilt = Eigen::Affine3f::Identity();
    tilt.rotate(Eigen::AngleAxisf(tilt_rad, Eigen::Vector3f::UnitY()));
    pcl::transformPointCloud(*cloud, *cloud, tilt);

    // 3) GPU CropBox
    GPUCloud d_in, d_roi;
    d_in.upload(cloud->points);
    gpu_cropbox.setMin(Eigen::Vector4f(p.roi_x_min, p.roi_y_min, p.roi_z_min, 1.0f));
    gpu_cropbox.setMax(Eigen::Vector4f(p.roi_x_max, p.roi_y_max, p.roi_z_max, 1.0f));
    gpu_cropbox.setInputCloud(d_in);
    gpu_cropbox.filter(d_roi);

    // 4) GPU VoxelGrid
    GPUCloud d_vox;
    gpu_voxel.setLeafSize(p.voxel_size, p.voxel_size, p.voxel_size);
    gpu_voxel.setInputCloud(d_roi);
    gpu_voxel.filter(d_vox);

    // 5) GPU SOR
    GPUCloud d_sor;
    gpu_sor.setInputCloud(d_vox);
    gpu_sor.setMeanK(50);
    gpu_sor.setStddevMulThresh(1.0);
    gpu_sor.filter(d_sor);

    // 6) CPU RANSAC 지면 제거
    CloudT::Ptr downsampled(new CloudT);
    downsampled->points.reserve(d_sor.size());
    d_sor.download(downsampled->points);

    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(false);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(p.ground_thresh);
    seg.setInputCloud(downsampled);
    seg.segment(*inliers, *coeff);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(downsampled);
    extract.setIndices(inliers);
    extract.setNegative(true);
    CloudT::Ptr non_ground(new CloudT);
    extract.filter(*non_ground);
    if (non_ground->empty()) return;

    // 7) 선택적 퍼블리시 (비지면)
    sensor_msgs::PointCloud2 ng_msg;
    pcl::toROSMsg(*non_ground, ng_msg);
    ng_msg.header = msg->header;
    clustered_pub.publish(ng_msg);

    // 8) GPU 클러스터링
    GPUCloud d_filtered;
    d_filtered.upload(non_ground->points);
    gpu_octree.setCloud(pcl::gpu::Octree::PointCloud(d_filtered));
    gpu_ec.setSearchMethod(gpu_octree);
    gpu_ec.setMinClusterSize(p.min_cluster_size);
    gpu_ec.setMaxClusterSize(p.max_cluster_size);
    gpu_ec.setClusterTolerance(p.base_tol + p.alpha_tol * 15.0f);
    std::vector<pcl::PointIndices> clusters;
    gpu_ec.extract(clusters);

    // 9) 후처리 & 중점 추출
    sensor_msgs::PointCloud pc;
    pc.header = msg->header;
    for (auto& idx : clusters) {
        CloudT cluster_pc; cluster_pc.reserve(idx.indices.size());
        for (int i : idx.indices) cluster_pc.points.push_back(non_ground->points[i]);

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(cluster_pc, min_pt, max_pt);
        float len = max_pt.x()-min_pt.x(), wid = max_pt.y()-min_pt.y();
        float height = max_pt.z()-min_pt.z();
        float diameter = std::max(len, wid);

        bool is_cone = height>=p.cone_h[0] && height<=p.cone_h[1]
                     && diameter>=p.cone_d[0] && diameter<=p.cone_d[1];
        bool is_drum = height>=p.drum_h[0] && height<=p.drum_h[1]
                     && diameter>=p.drum_d[0] && diameter<=p.drum_d[1];
        if (!is_cone && !is_drum) continue;

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(cluster_pc, centroid);
        geometry_msgs::Point32 pt;
        pt.x=centroid[0]; pt.y=centroid[1]; pt.z=0;
        pc.points.push_back(pt);
    }
    static_pub.publish(pc);

    // 10) 시간 측정 종료 및 로깅
    ros::Time end = ros::Time::now();
    ROS_INFO_STREAM("cloudCallback duration: " << (end-start).toSec()*1000 << " ms");
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "livox_static_gpu");
    ros::NodeHandle nh;
    clustered_pub = nh.advertise<sensor_msgs::PointCloud2>("/clustered_points", 1);
    static_pub    = nh.advertise<sensor_msgs::PointCloud> ("/static_obstacles",1);
    ros::Subscriber sub = nh.subscribe("/livox/lidar", 1, cloudCallback);
    ros::spin();
    return 0;
}
