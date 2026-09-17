#!/usr/bin/env python3
# scripts/livox_obstacle_detector.py

import rospy
import numpy as np
try:
    import cupy as cp
    xp = cp
    use_cuda = True
    rospy.loginfo("CUDA acceleration enabled (CuPy)")
except ImportError:
    xp = np
    use_cuda = False
    rospy.logwarn("CuPy not found; falling back to NumPy")

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from sklearn.linear_model import RANSACRegressor
from hdbscan import HDBSCAN
import std_msgs.msg

#--- 설정 파라미터 ---------------------------------------
ROI = {"x": (-10, 10), "y": (-5, 5), "z": (-5, 5)}
VOXEL_SIZE = 0.1
GROUND_THRESH = 0.2
TILT_ANGLE_DEG = 3.3
TILT_RAD = np.deg2rad(TILT_ANGLE_DEG)
CONE_H = (0.5, 1.0); CONE_D = (0.2, 0.5)
DRUM_H = (0.8, 1.3); DRUM_D = (0.4, 1.0)
MERGE_GAP = 1.0
MAX_LEN = 5.0; MAX_WID = 2.5
#--------------------------------------------------------

static_pub     = None  # MarkerArray for static obstacles
clustered_pub  = None  # PointCloud2 for clustered points
prev_marker_ids = set()

def rotate_tilt(points):
    cos_t, sin_t = np.cos(TILT_RAD), np.sin(TILT_RAD)
    x = points[:,0:1]; y = points[:,1:2]; z = points[:,2:3]
    xr = x*cos_t - z*sin_t
    zr = x*sin_t + z*cos_t
    return xp.hstack((xr, y, zr))

def voxel_downsample(points, size=VOXEL_SIZE):
    discrete = xp.floor(points / size)
    _, idx = xp.unique(discrete, axis=0, return_index=True)
    return points[idx]

def remove_ground_ransac(points, thresh=GROUND_THRESH):
    X = points[:, :2].get() if use_cuda else points[:, :2]
    y = points[:, 2].get() if use_cuda else points[:, 2]
    ransac = RANSACRegressor(residual_threshold=thresh)
    try:
        ransac.fit(X, y)
        pred = ransac.predict(X)
        mask = np.abs(y - pred) > thresh
        return points[mask]
    except Exception as e:
        rospy.logwarn(f"RANSAC failed: {e}")
        return points

def merge_close_clusters(points, labels, gap=MERGE_GAP):
    unique = set(labels) - {-1}
    centroids = {l: xp.mean(points[labels==l], axis=0) for l in unique}
    mapping, merged = {}, set()
    for l1 in unique:
        if l1 in merged: continue
        mapping[l1] = l1
        for l2 in unique:
            if l2 == l1 or l2 in merged: continue
            if xp.linalg.norm(centroids[l1][:2] - centroids[l2][:2]) < gap:
                mapping[l2] = l1; merged.add(l2)
    return xp.array([mapping.get(l, -1) if l != -1 else -1 for l in labels])

def filter_by_bbox(cluster, max_len=MAX_LEN, max_wid=MAX_WID):
    pts = cluster.get() if use_cuda else cluster
    return (np.ptp(pts[:,0]) < max_len) and (np.ptp(pts[:,1]) < max_wid)

def classify_cluster(cluster):
    pts = cluster.get() if use_cuda else cluster
    height = pts[:,2].max() - pts[:,2].min()
    diameter = max(np.ptp(pts[:,0]), np.ptp(pts[:,1]))
    if CONE_H[0] <= height <= CONE_H[1] and CONE_D[0] <= diameter <= CONE_D[1]: return "cone"
    if DRUM_H[0] <= height <= DRUM_H[1] and DRUM_D[0] <= diameter <= DRUM_D[1]: return "drum"
    return None

def pointcloud_callback(msg):
    global prev_marker_ids
    pts_list = list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True))
    if not pts_list: return
    pts = xp.array(pts_list, dtype=xp.float32)

    pts = rotate_tilt(pts)
    mask = ((ROI["x"][0] <= pts[:,0]) & (pts[:,0] <= ROI["x"][1]) &
            (ROI["y"][0] <= pts[:,1]) & (pts[:,1] <= ROI["y"][1]) &
            (ROI["z"][0] <= pts[:,2]) & (pts[:,2] <= ROI["z"][1]))
    pts = pts[mask]
    if pts.size == 0: return

    ds = voxel_downsample(pts)
    ng = remove_ground_ransac(ds)
    hdr = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=msg.header.frame_id)
    clustered_pub.publish(pc2.create_cloud_xyz32(hdr, (ng.get() if use_cuda else ng)))
    if ng.size == 0: return

    cpu_pts = ng.get() if use_cuda else ng
    labels0 = HDBSCAN(min_cluster_size=5, min_samples=3,
                      cluster_selection_epsilon=1.5).fit_predict(cpu_pts)
    labels = merge_close_clusters(ng, labels0)

    marker_array = MarkerArray()
    curr_ids = set()
    mid = 0
    for lbl in set(labels):
        if lbl == -1: continue
        cluster = ng[labels == lbl]
        if not filter_by_bbox(cluster): continue
        cls = classify_cluster(cluster)
        if not cls: continue

        cen = xp.mean(cluster, axis=0)
        cx, cy, cz = (cen.get() if use_cuda else cen)

        m = Marker()
        m.header.frame_id = msg.header.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "static_obstacles"
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(cx), float(cy), float(cz)
        m.scale.x = m.scale.y = m.scale.z = 0.3
        if cls == "cone":
            m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0
        else:
            m.color.r, m.color.g, m.color.b = 0.0, 0.0, 1.0
        m.color.a = 0.8
        marker_array.markers.append(m)
        curr_ids.add(mid)
        mid += 1

    for did in prev_marker_ids - curr_ids:
        dm = Marker()
        dm.header.frame_id = msg.header.frame_id
        dm.header.stamp = rospy.Time.now()
        dm.ns = "static_obstacles"
        dm.id = did
        dm.action = Marker.DELETE
        marker_array.markers.append(dm)
    prev_marker_ids = curr_ids

    static_pub.publish(marker_array)

if __name__ == "__main__":
    rospy.init_node("livox_obstacle_detector")
    static_pub    = rospy.Publisher("/static_obstacles", MarkerArray, queue_size=1)
    clustered_pub = rospy.Publisher("/clustered_points", PointCloud2, queue_size=1)
    rospy.Subscriber("/livox/lidar", PointCloud2, pointcloud_callback, queue_size=1)
    rospy.spin()
