#!/usr/bin/env python3
# ===============================================================
#  Velodyne PointCloud → BEV → YOLO v8 검출 → OC-SORT 추적
#  (noahcao/OC_SORT 버전용, img_info 인자 포함)
# ===============================================================

# ====================== 파라미터 블록 ==========================
MODEL_PATH          = "/home/cnu/clothoid-r/perception_ws/src/lidar_object_detec/model/velodyne_v6.pt"
VOXEL_SIZE          = 0.05
X_RANGE             = (-15.0, 15.0)
Y_RANGE             = (-15.0, 15.0)
Z_RANGE             = (-2.5,  2.0)
MAX_PTS_PER_VOXEL   = 30
DETECT_CONF         = 0.3
OCSORT_IOU_THRESH   = 0.25
OCSORT_MAX_AGE      = 10
OCSORT_MIN_HITS     = 3
OCSORT_DELTA_T      = 1
HEARTBEAT_HZ        = 10
# ===============================================================

import os, sys, rospy, numpy as np, cv2
from sensor_msgs.msg import PointCloud2, PointCloud, Image
from geometry_msgs.msg import Point32
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO

sys.path.insert(0, "/opt/OC_SORT")          # OC_SORT clone 경로
from trackers.ocsort_tracker.ocsort import OCSort

class BEV_OCSort_Node:
    def __init__(self):
        rospy.init_node("velodyne_bev_ocsort_node", anonymous=True)
        rospy.loginfo("▶︎ Velodyne BEV + YOLO + OC-SORT Tracker 초기화 중…")

        self.detector = YOLO(rospy.get_param("model_path", MODEL_PATH))

        self.voxel   = rospy.get_param("voxel_size", VOXEL_SIZE)
        self.x_rng   = tuple(rospy.get_param("x_range", list(X_RANGE)))
        self.y_rng   = tuple(rospy.get_param("y_range", list(Y_RANGE)))
        self.z_rng   = tuple(rospy.get_param("z_range", list(Z_RANGE)))
        self.max_pts = rospy.get_param("max_points_per_voxel", MAX_PTS_PER_VOXEL)

        self.img_h = int((self.x_rng[1] - self.x_rng[0]) / self.voxel)
        self.img_w = int((self.y_rng[1] - self.y_rng[0]) / self.voxel)

        self.tracker = OCSort(det_thresh    = DETECT_CONF,
                              iou_threshold = OCSORT_IOU_THRESH,
                              max_age       = OCSORT_MAX_AGE,
                              min_hits      = OCSORT_MIN_HITS,
                              delta_t       = OCSORT_DELTA_T)

        self.bridge = CvBridge()
        rospy.Subscriber("/velodyne_points", PointCloud2, self.lidar_cb, queue_size=1)
        self.pcl_pub    = rospy.Publisher("/detected_objects", PointCloud, queue_size=10)
        self.marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)
        self.img_pub    = rospy.Publisher("/bev_detection_image", Image, queue_size=10)

        self.last_msg = PointCloud(); self.last_msg.header.frame_id = "velodyne"
        rospy.Timer(rospy.Duration(1.0 / HEARTBEAT_HZ), self.timer_cb)

    def timer_cb(self, _):
        self.last_msg.header.stamp = rospy.Time.now()
        self.pcl_pub.publish(self.last_msg)

    # ---------- BEV 변환 함수(변경 없음) --------------------------
    def pc2_to_bev(self, pts):
        if not pts.size:
            return np.zeros((self.img_h, self.img_w, 3), np.uint8)
        x, y, z, inten = pts.T
        m = ((self.x_rng[0] <= x) & (x < self.x_rng[1]) &
             (self.y_rng[0] <= y) & (y < self.y_rng[1]) &
             (self.z_rng[0] <= z) & (z < self.z_rng[1]))
        x, y, z, inten = x[m], y[m], z[m], inten[m]
        ix = ((self.y_rng[1] - y) / self.voxel).astype(np.int32)
        iy = ((self.x_rng[1] - x) / self.voxel).astype(np.int32)
        ix = np.clip(ix, 0, self.img_w - 1); iy = np.clip(iy, 0, self.img_h - 1)
        hmap = np.full((self.img_h, self.img_w), self.z_rng[0], np.float32)
        imap = np.zeros_like(hmap); dmap = np.zeros_like(hmap, np.int32)
        np.maximum.at(hmap, (iy, ix), z)
        np.add.at(imap, (iy, ix), inten)
        np.add.at(dmap, (iy, ix), 1)
        h = ((np.clip(hmap, *self.z_rng) - self.z_rng[0]) /
             (self.z_rng[1] - self.z_rng[0]) * 255).astype(np.uint8)
        i = np.zeros_like(imap, np.uint8); i[dmap>0] = np.clip(imap[dmap>0]/dmap[dmap>0],0,255).astype(np.uint8)
        d = (np.clip(dmap,0,self.max_pts)/self.max_pts*255).astype(np.uint8)
        return cv2.merge([d,i,h])

    # ---------- LiDAR 콜백 --------------------------------------
    def lidar_cb(self, msg: PointCloud2):
        use_i = any(f.name == 'intensity' for f in msg.fields)
        fields = ('x','y','z','intensity') if use_i else ('x','y','z')
        pts = np.asarray([p for p in pc2.read_points(msg, field_names=fields, skip_nans=True)],
                         dtype=np.float32)
        if not use_i and pts.size:
            pts = np.hstack([pts, np.zeros((pts.shape[0],1), np.float32)])

        bev = self.pc2_to_bev(pts)
        vis = bev.copy()

        dets = np.asarray([[*b.xyxy[0].cpu().numpy(), float(b.conf[0])]
                           for b in self.detector.predict(bev, device='cuda',
                                                          conf=DETECT_CONF)[0].boxes],
                          dtype=np.float32)
        if dets.size == 0:
            dets = np.empty((0,5), np.float32)

        # --- OC-SORT 업데이트 (img_info, img_size 인자 순서 주의) ---
        tracks = self.tracker.update(dets,
                                     (self.img_h, self.img_w),   # img_info
                                     (self.img_h, self.img_w))   # img_size

        pcl_msg = PointCloud(); pcl_msg.header.frame_id = "velodyne"; pcl_msg.header.stamp = rospy.Time.now()
        m_arr = MarkerArray()

        for x1,y1,x2,y2,tid in tracks:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            lx = self.x_rng[0] + (self.img_h-1-cy)*self.voxel
            ly = self.y_rng[0] + (self.img_w-1-cx)*self.voxel
            pcl_msg.points.append(Point32(lx,ly,0.0))

            mk = Marker(); mk.header = pcl_msg.header
            mk.ns, mk.id = "track", int(tid); mk.type = Marker.TEXT_VIEW_FACING
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = lx, ly, 0.6
            mk.text = f"{int(tid)}"; mk.scale.z = 0.45
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1,1,0,1
            m_arr.markers.append(mk)

            cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            cv2.putText(vis,f"ID{int(tid)}",(int(x1),int(y1)-7),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)

        self.pcl_pub.publish(pcl_msg)
        self.marker_pub.publish(m_arr)
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(vis,"bgr8"))
        self.last_msg = pcl_msg

# ---------------------------- main ----------------------------- #
if __name__ == "__main__":
    BEV_OCSort_Node()
    rospy.loginfo("▶︎ 노드 실행 중… /detected_objects 10 Hz heartbeat 보장")
    rospy.spin()
