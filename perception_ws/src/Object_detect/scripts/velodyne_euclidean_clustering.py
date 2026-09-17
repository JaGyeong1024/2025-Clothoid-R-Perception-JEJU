#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR Pre-processing + Distance-weighted Euclidean Clustering + Kalman Tracking
(출력: /preprocessed_points, /cluster_centroids)
ROS Noetic 기준
"""

import rospy, time, numpy as np
from sensor_msgs.msg import PointCloud2, PointCloud
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Point32
from scipy.spatial import cKDTree
import std_msgs.msg

# -------------------- 하드코딩 파라미터 --------------------
# 1) 센서 / ROI / 다운샘플
PITCH_DEG        = 0.001
ROI_X_MIN, ROI_X_MAX = 12, 12
ROI_Y_MIN, ROI_Y_MAX = -4, 4
ROI_Z_MIN, ROI_Z_MAX = -2, 1.5
VOXEL_SIZE       = 0.01

# 2) DROR 외란 제거
DROR_MIN_NEIGHBORS = 1
DROR_MIN_RADIUS    = 0.1
DROR_RADIUS_SCALE  = 0.1
DROR_MAX_RADIUS    = 0.2

# 3) 그리드 기반 지면 제거
GRID_CELL_SIZE        = 0.2   # [m]
GRID_MAX_HEIGHT_DIFF  = 0.2   # [m]
GRID_MIN_POINTS       = 10

# 4) RANSAC 잔류 지면 제거
GROUND_THRESH = 0.3

# 5) 클러스터링 / 바운딩박스
EUCLIDEAN_MIN_CLUSTER_SIZE = 5
CLUSTER_MERGE_GAP          = 0.5
MAX_LENGTH, MAX_WIDTH, MAX_HEIGHT = 2.5, 2.5, 1.0
MIN_LENGTH, MIN_WIDTH, MIN_HEIGHT = 0.5, 0.5, 0.3 # 0.1 # 0.3 <- erp 위주
EUCLIDEAN_BASE_DIST, EUCLIDEAN_DIST_SCALE = 0.05, 0.05 #0.08 0.012

# 6) 트래킹
TRACKER_MAX_MISS = 5
MATCH_DIST       = 1.5
# ---------------------------------------------------------

# ---------- 보조 클래스 ----------
class KalmanFilter:
    def __init__(self, dt=0.1, q=5.0, r=0.1):
        self.A = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])
        self.H = np.array([[1,0,0,0],  [0,1,0,0]])
        self.Q = q*np.eye(4)
        self.R = r*np.eye(2)
        self.P = np.eye(4)
        self.x = np.zeros((4,1))
    def predict(self):
        self.x = self.A@self.x
        self.P = self.A@self.P@self.A.T + self.Q
        return self.x[:2].flatten()
    def update(self, z):
        z = np.reshape(z,(2,1))
        y = z - self.H@self.x
        S = self.H@self.P@self.H.T + self.R
        K = self.P@self.H.T@np.linalg.inv(S)
        self.x += K@y
        self.P = (np.eye(4)-K@self.H)@self.P

class Tracker:
    def __init__(self, c, tid, dt=0.1):
        self.id = tid; self.kf = KalmanFilter(dt); self.kf.update(c)
        self.miss = 0; self.last = c
    def predict(self):
        self.last = self.kf.predict(); return self.last
    def update(self, c): self.kf.update(c); self.last=c; self.miss=0
    def no_update(self): self.kf.predict(); self.miss+=1

# ---------- 유틸리티 ----------
def rot_pitch(points, deg):
    th = np.deg2rad(deg)
    R = np.array([[ np.cos(th),0, np.sin(th)],
                  [ 0,          1, 0        ],
                  [-np.sin(th),0, np.cos(th)]])
    return points @ R.T

def voxel_downsample(pts, vs):
    if len(pts)==0: return pts
    idx = np.unique(np.floor(pts/vs), axis=0, return_index=True)[1]
    return pts[idx]

def dror_filter(pts):
    if len(pts)==0: return pts
    rng = np.linalg.norm(pts[:,:2],axis=1)
    radii = np.clip(DROR_MIN_RADIUS + DROR_RADIUS_SCALE*rng,
                    DROR_MIN_RADIUS, DROR_MAX_RADIUS)
    tree = cKDTree(pts)
    keep = [len(tree.query_ball_point(pts[i],radii[i]))-1 >= DROR_MIN_NEIGHBORS
            for i in range(len(pts))]
    return pts[np.array(keep)]

def grid_ground_remove(pts):
    if len(pts)==0: return pts
    ix = np.floor((pts[:,0]-ROI_X_MIN)/GRID_CELL_SIZE).astype(int)
    iy = np.floor((pts[:,1]-ROI_Y_MIN)/GRID_CELL_SIZE).astype(int)
    key = ix*100000 + iy
    min_z, cnt = {}, {}
    for k,z in zip(key, pts[:,2]):
        if k in min_z:
            if z<min_z[k]: min_z[k]=z
            cnt[k]+=1
        else: min_z[k]=z; cnt[k]=1
    ground = [(cnt[k]>=GRID_MIN_POINTS and pts[i,2]-min_z[k]<GRID_MAX_HEIGHT_DIFF)
              for i,k in enumerate(key)]
    return pts[~np.array(ground)]

def ransac_ground_remove(pts):
    if len(pts)==0: return pts
    X, y = pts[:,:2], pts[:,2]
    from sklearn.linear_model import RANSACRegressor
    try:
        ransac = RANSACRegressor(residual_threshold=GROUND_THRESH).fit(X,y)
        res = np.abs(y - ransac.predict(X))
        return pts[res>GROUND_THRESH]
    except: return pts

def dist_euclid_labels(pts):
    if len(pts)==0: return np.array([],int)
    tree = cKDTree(pts[:,:2])
    n = len(pts)
    visited = np.zeros(n,bool); lbl = -np.ones(n,int); cid=0
    for i in range(n):
        if visited[i]: continue
        d0 = EUCLIDEAN_BASE_DIST + EUCLIDEAN_DIST_SCALE*abs(pts[i,0])
        Q = tree.query_ball_point(pts[i,:2],d0)
        if len(Q) < EUCLIDEAN_MIN_CLUSTER_SIZE:
            visited[i]=True; continue
        stack = list(Q); lbl[stack]=cid; visited[stack]=True
        while stack:
            cur=stack.pop()
            d = EUCLIDEAN_BASE_DIST+EUCLIDEAN_DIST_SCALE*abs(pts[cur,0])
            for nb in tree.query_ball_point(pts[cur,:2],d):
                if not visited[nb]:
                    visited[nb]=True
                    if len(tree.query_ball_point(pts[nb,:2],d))>=EUCLIDEAN_MIN_CLUSTER_SIZE:
                        stack.append(nb)
                    lbl[nb]=cid
        cid+=1
    return lbl

def merge_clusters(pts, lbl):
    uniq=set(lbl); uniq.discard(-1)
    if not uniq: return lbl
    cent={l:np.mean(pts[lbl==l],axis=0) for l in uniq}
    rep={}; merged=set()
    for l1 in uniq:
        if l1 in merged: continue
        rep[l1]=l1
        for l2 in uniq:
            if l1==l2 or l2 in merged: continue
            if np.linalg.norm(cent[l1][:2]-cent[l2][:2])<CLUSTER_MERGE_GAP:
                rep[l2]=l1; merged.add(l2)
    return np.array([rep.get(x,-1) if x!=-1 else -1 for x in lbl])

def bbox_ok(pts):
    if len(pts)==0: return False
    xl,yl,zl = np.ptp(pts[:,0]), np.ptp(pts[:,1]), np.ptp(pts[:,2])
    return (MIN_LENGTH<xl<MAX_LENGTH and
            MIN_WIDTH <yl<MAX_WIDTH  and
            MIN_HEIGHT<zl<MAX_HEIGHT)

# ---------- 전역 상태 ----------
trackers, tid_seq = {}, 0

# ---------- ROS Publisher ----------
pre_pub   = rospy.Publisher("/preprocessed_points", PointCloud2, queue_size=1)
cent_pub  = rospy.Publisher("/jagyeong",  PointCloud,   queue_size=1)

def publish_cloud(pub, pts):
    hdr = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="livox_frame")
    pub.publish(pc2.create_cloud_xyz32(hdr, pts))

def publish_centroids(c_list):
    pc_msg = PointCloud()
    pc_msg.header.stamp = rospy.Time.now()
    pc_msg.header.frame_id = "livox_frame"
    pc_msg.points = [Point32(x,y,0.0) for x,y in c_list]
    cent_pub.publish(pc_msg)

# ---------- 메인 콜백 ----------
def pc_callback(msg):
    global trackers, tid_seq
    start = time.time()

    # (1) Point read
    pts = np.array([[p[0],p[1],p[2]] for p in
                    pc2.read_points(msg,field_names=("x","y","z"),skip_nans=True)])
    if pts.size==0:
        publish_cloud(pre_pub, [])      # 빈 PointCloud2
        publish_centroids([])           # 빈 centroid
        return

    # (2) Pitch 보정 + ROI
    pts = rot_pitch(pts, PITCH_DEG)
    mask = ((ROI_X_MIN<=pts[:,0])&(pts[:,0]<=ROI_X_MAX)&
            (ROI_Y_MIN<=pts[:,1])&(pts[:,1]<=ROI_Y_MAX)&
            (ROI_Z_MIN<=pts[:,2])&(pts[:,2]<=ROI_Z_MAX))
    pts = pts[mask]

    # (3) 다운샘플·DROR
    pts = dror_filter(voxel_downsample(pts, VOXEL_SIZE))

    # (4) 지면 제거 (그리드 → RANSAC)
    pts = ransac_ground_remove(grid_ground_remove(pts))

    # (5) /preprocessed_points 발행 (빈 배열이라도)
    publish_cloud(pre_pub, pts)

    # (6) 클러스터링
    lbl = merge_clusters(pts, dist_euclid_labels(pts))
    observed = [np.mean(pts[lbl==cid][:,:2],axis=0)
                for cid in set(lbl) if cid!=-1 and bbox_ok(pts[lbl==cid])]

    observed = np.asarray(observed, dtype=float)

    # (7) 트래커 매칭
    preds = np.array([t.predict() for t in trackers.values()]) if trackers else np.zeros((0,2))
    keys = list(trackers.keys())
    assigned_trk, assigned_obs = set(), set()
    if len(observed) and len(preds):
        dist = np.linalg.norm(observed[:,None,:]-preds[None,:,:],axis=2)
        while True:
            i,j = np.unravel_index(np.argmin(dist), dist.shape)
            if dist[i,j] > MATCH_DIST: break
            trackers[keys[j]].update(observed[i])
            assigned_trk.add(keys[j]); assigned_obs.add(i)
            dist[i,:]=dist[:,j]=np.inf
            if np.isinf(dist).all(): break
    for k,t in list(trackers.items()):
        if k not in assigned_trk:
            t.no_update()
            if t.miss > TRACKER_MAX_MISS: del trackers[k]
    for i,c in enumerate(observed):
        if i not in assigned_obs:
            trackers[tid_seq] = Tracker(c, tid_seq); tid_seq+=1

    # (8) /cluster_centroids 발행 (트래커 기준)
    publish_centroids([t.last for t in trackers.values()])

    rospy.logdebug(f"callback {(time.time()-start):.3f}s")

# ---------- 노드 초기화 ----------
if __name__ == "__main__":
    rospy.init_node("lidar_clustering_simple")
    rospy.Subscriber("/velodyne", PointCloud2, pc_callback, queue_size=1)
    rospy.spin()
