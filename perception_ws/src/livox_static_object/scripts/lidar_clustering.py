#!/usr/bin/env python3
import rospy
import numpy as np
import time
from sensor_msgs.msg import PointCloud2, PointCloud
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Point32
from sklearn.linear_model import RANSACRegressor
import hdbscan
import std_msgs.msg
from scipy.spatial import cKDTree

# -------------------- 파라미터 영역 --------------------
ROI_X_MIN = 0
ROI_X_MAX = 25
ROI_Y_MIN = -1.2
ROI_Y_MAX = 1.2
ROI_Z_MIN = -2
ROI_Z_MAX = 2

VOXEL_SIZE = 0.05   # 크면 잡음↓, 연산량↓, 구조 손실↑ / 작으면 구조 보존↑, 잡음↑, 연산량↑

DROR_MIN_NEIGHBORS = 3    # 크면 외란에 강함, 작으면 작은 장애물 검출↑
DROR_MIN_RADIUS = 0.15    # 크면 노이즈 제거↑, 객체 연속성↓
DROR_RADIUS_SCALE = 0.06  # 크면 먼 거리 노이즈 제거↑, 작은 객체 손실↑
DROR_MAX_RADIUS = 0.1     # 크면 먼 거리 노이즈 제거↑, 작으면 먼 거리 작은 물체 검출↑

GROUND_THRESH = 0.4       # 크면 지면 인식 범위↑(경사로 포함), 작으면 평지에만 적용

HDBSCAN_MIN_CLUSTER_SIZE = 4   # 크면 잡음 무시↑, 작은 객체 무시↑, 작으면 작은 객체 검출↑
HDBSCAN_MIN_SAMPLES = 3        # 크면 외란에 강함, 작으면 민감
HDBSCAN_EPSILON = 0.3          # 크면 클러스터 병합↑, 작으면 세분화↑

CLUSTER_MERGE_GAP = 0.1        # 크면 인접 클러스터 병합↑, 작으면 분리↑

MAX_LENGTH = 2.0
MAX_WIDTH = 2.0
MAX_HEIGHT = 1.5
MIN_LENGTH = 0.1
MIN_WIDTH = 0.1
MIN_HEIGHT = 0.1
# ------------------------------------------------------

class KalmanFilter:
    def __init__(self, dt=0.1, process_noise=5.0, measurement_noise=0.1):
        self.dt = dt
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = process_noise * np.eye(4)
        self.R = measurement_noise * np.eye(2)
        self.P = np.eye(4)
        self.x = np.zeros((4, 1))

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        z = np.reshape(z, (2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

ROI = {
    "x": (ROI_X_MIN, ROI_X_MAX),
    "y": (ROI_Y_MIN, ROI_Y_MAX),
    "z": (ROI_Z_MIN, ROI_Z_MAX)
}

prev_marker_ids = set()
tracker_dict = {}
last_callback_time = [None]

def voxel_downsample(points, voxel_size=VOXEL_SIZE):
    discrete = np.floor(points / voxel_size)
    _, idx = np.unique(discrete, axis=0, return_index=True)
    return points[idx]

def dror(points, min_neighbors=DROR_MIN_NEIGHBORS, min_radius=DROR_MIN_RADIUS, radius_scale=DROR_RADIUS_SCALE, max_radius=DROR_MAX_RADIUS):
    if len(points) == 0:
        return points
    xy = points[:, :2]
    ranges = np.linalg.norm(xy, axis=1)
    radii = np.clip(min_radius + radius_scale * ranges, min_radius, max_radius)
    tree = cKDTree(points)
    neighbor_counts = np.array([len(tree.query_ball_point(points[i], radii[i])) - 1 for i in range(len(points))])
    mask = neighbor_counts >= min_neighbors
    return points[mask]

def remove_ground_ransac(points, threshold=GROUND_THRESH):
    X = points[:, :2]
    y = points[:, 2]
    ransac = RANSACRegressor(residual_threshold=threshold)
    try:
        ransac.fit(X, y)
        z_pred = ransac.predict(X)
        residuals = np.abs(y - z_pred)
        mask = residuals > threshold
        return points[mask]
    except:
        return points

def filter_by_bounding_box(cluster_points, max_length=MAX_LENGTH, max_width=MAX_WIDTH, max_height=MAX_HEIGHT, min_length=MIN_LENGTH, min_width=MIN_WIDTH, min_height=MIN_HEIGHT):
    x_len = np.ptp(cluster_points[:, 0])
    y_len = np.ptp(cluster_points[:, 1])
    z_len = np.ptp(cluster_points[:, 2])
    return (min_length < x_len < max_length) and (min_width < y_len < max_width) and (min_height < z_len < max_height)

def publish_downsampled_points(points, frame_id):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    cloud_msg = pc2.create_cloud_xyz32(header, points)
    downsample_pub.publish(cloud_msg)

def merge_close_clusters(points, labels, max_gap=CLUSTER_MERGE_GAP):
    unique_labels = set(labels)
    unique_labels.discard(-1)
    centroids = {l: np.mean(points[labels == l], axis=0) for l in unique_labels}
    merged_labels = dict()
    merged = set()
    for l1 in unique_labels:
        if l1 in merged:
            continue
        merged_labels[l1] = l1
        for l2 in unique_labels:
            if l1 == l2 or l2 in merged:
                continue
            dist = np.linalg.norm(centroids[l1][:2] - centroids[l2][:2])
            if dist < max_gap:
                merged_labels[l2] = l1
                merged.add(l2)
    new_labels = np.array([merged_labels.get(lbl, -1) if lbl != -1 else -1 for lbl in labels])
    return new_labels

def compute_obb_marker(cluster_points, cluster_id, frame_id):
    points_2d = cluster_points[:, :2]
    centroid = np.mean(points_2d, axis=0)
    centered = points_2d - centroid
    cov = np.cov(centered.T)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    order = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, order]
    rotated = centered @ eig_vecs
    min_xy = np.min(rotated, axis=0)
    max_xy = np.max(rotated, axis=0)
    size = max_xy - min_xy
    center_local = (max_xy + min_xy) / 2.0
    obb_center = centroid + eig_vecs @ center_local
    yaw = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
    qz = np.sin(yaw / 2.0)
    qw = np.cos(yaw / 2.0)
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "obb_boxes"
    marker.id = cluster_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = obb_center[0]
    marker.pose.position.y = obb_center[1]
    marker.pose.position.z = np.mean(cluster_points[:, 2])
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw
    marker.scale.x = size[0]
    marker.scale.y = size[1]
    marker.scale.z = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.6
    return marker

def pointcloud_callback(msg):
    global prev_marker_ids, tracker_dict

    now = time.time()
    if last_callback_time[0] is not None:
        print(f"[콜백 간격] {now - last_callback_time[0]:.3f}초")
    last_callback_time[0] = now

    t_total_start = time.time()

    points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True)])
    if len(points) == 0:
        print("입력 포인트 없음")
        return

    x_cond = (ROI["x"][0] <= points[:, 0]) & (points[:, 0] <= ROI["x"][1])
    y_cond = (ROI["y"][0] <= points[:, 1]) & (points[:, 1] <= ROI["y"][1])
    z_cond = (ROI["z"][0] <= points[:, 2]) & (points[:, 2] <= ROI["z"][1])
    roi_points = points[x_cond & y_cond & z_cond]
    if len(roi_points) == 0:
        print("ROI 통과 포인트 없음")
        return

    downsampled_points = voxel_downsample(roi_points, voxel_size=VOXEL_SIZE)
    if len(downsampled_points) == 0:
        print("다운샘플 포인트 없음")
        return

    dror_points = dror(downsampled_points,
                       min_neighbors=DROR_MIN_NEIGHBORS,
                       min_radius=DROR_MIN_RADIUS,
                       radius_scale=DROR_RADIUS_SCALE,
                       max_radius=DROR_MAX_RADIUS)
    if len(dror_points) == 0:
        print("DROR 후 포인트 없음")
        return

    non_ground_points = remove_ground_ransac(dror_points, threshold=GROUND_THRESH)
    publish_downsampled_points(non_ground_points, msg.header.frame_id)
    if len(non_ground_points) == 0:
        print("지면 제거 후 포인트 없음")
        return

    t_hdbscan_start = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                min_samples=HDBSCAN_MIN_SAMPLES,
                                cluster_selection_epsilon=HDBSCAN_EPSILON)
    labels = clusterer.fit_predict(non_ground_points)
    t_hdbscan_end = time.time()
    print(f"[HDBSCAN 수행시간] {t_hdbscan_end - t_hdbscan_start:.3f}초, 입력 포인트 {len(non_ground_points)}개")

    labels = merge_close_clusters(non_ground_points, labels, max_gap=CLUSTER_MERGE_GAP)

    marker_array = MarkerArray()
    curr_marker_ids = set()
    new_tracker_dict = {}

    cluster_pos_markers = MarkerArray()

    # ------ Centroid PointCloud용 ------
    centroid_points = []

    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        cluster_points = non_ground_points[labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        if not filter_by_bounding_box(cluster_points):
            continue

        centroid = np.mean(cluster_points[:, :2], axis=0)
        centroid_points.append(Point32(centroid[0], centroid[1], 0.0))

        tracker = tracker_dict.get(cluster_id, KalmanFilter())
        tracker.predict()
        tracker.update(centroid)
        tracked_pos = tracker.x[:2].flatten()

        pos_marker = Marker()
        pos_marker.header.frame_id = msg.header.frame_id
        pos_marker.header.stamp = rospy.Time.now()
        pos_marker.ns = "cluster_positions"
        pos_marker.id = cluster_id
        pos_marker.type = Marker.SPHERE
        pos_marker.action = Marker.ADD
        pos_marker.pose.position.x = tracked_pos[0]
        pos_marker.pose.position.y = tracked_pos[1]
        pos_marker.pose.position.z = np.mean(cluster_points[:, 2])
        pos_marker.scale.x = 0.3
        pos_marker.scale.y = 0.3
        pos_marker.scale.z = 0.3
        pos_marker.color.r = 1.0
        pos_marker.color.g = 0.0
        pos_marker.color.b = 0.0
        pos_marker.color.a = 0.8
        cluster_pos_markers.markers.append(pos_marker)

        marker = compute_obb_marker(cluster_points, cluster_id, msg.header.frame_id)
        marker_array.markers.append(marker)

        new_tracker_dict[cluster_id] = tracker
        curr_marker_ids.add(cluster_id)

    removed_ids = prev_marker_ids - curr_marker_ids
    for rem_id in removed_ids:
        del_marker = Marker()
        del_marker.header.frame_id = msg.header.frame_id
        del_marker.header.stamp = rospy.Time.now()
        del_marker.ns = "obb_boxes"
        del_marker.id = rem_id
        del_marker.action = Marker.DELETE
        marker_array.markers.append(del_marker)

        del_pos_marker = Marker()
        del_pos_marker.header.frame_id = msg.header.frame_id
        del_pos_marker.header.stamp = rospy.Time.now()
        del_pos_marker.ns = "cluster_positions"
        del_pos_marker.id = rem_id
        del_pos_marker.action = Marker.DELETE
        cluster_pos_markers.markers.append(del_pos_marker)

    prev_marker_ids = curr_marker_ids
    tracker_dict = new_tracker_dict

    marker_pub.publish(marker_array)
    cluster_pos_pub.publish(cluster_pos_markers)

    # ----- Centroid PointCloud Publish -----
    pc_msg = PointCloud()
    pc_msg.header.stamp = rospy.Time.now()
    pc_msg.header.frame_id = msg.header.frame_id
    pc_msg.points = centroid_points
    centroid_pub.publish(pc_msg)

    t_total_end = time.time()
    print(f"[콜백 전체 처리시간] {t_total_end - t_total_start:.3f}초\n")

if __name__ == "__main__":
    rospy.init_node("lidar_clustering_node")

    rospy.Subscriber("/livox/lidar", PointCloud2, pointcloud_callback)
    marker_pub = rospy.Publisher("/lidar_clusters", MarkerArray, queue_size=1)
    cluster_pos_pub = rospy.Publisher("/cluster_positions", MarkerArray, queue_size=10)
    downsample_pub = rospy.Publisher("/downsampled_points", PointCloud2, queue_size=1)
    centroid_pub = rospy.Publisher("/cluster_centroids", PointCloud, queue_size=1)

    rospy.spin()
