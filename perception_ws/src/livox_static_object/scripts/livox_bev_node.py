#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from cv_bridge import CvBridge

class LivoxBevNode:
    def __init__(self):
        rospy.init_node('livox_bev_node', anonymous=True)
        self.sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.lidar_callback, queue_size=1)
        self.pub = rospy.Publisher('/livox/bev', Image, queue_size=1)
        self.bridge = CvBridge()

        # ===== BEV 파라미터 (코드 내에서 직접 지정) =====
        self.bev_width = 512
        self.bev_height = 512
        self.x_min = 0.0
        self.x_max = 30.0
        self.y_min = -10.0
        self.y_max = 10.0
        self.z_min = -0.5
        self.z_max = 0.1

        # ===== pitch_deg: 라이다 pitch 보정 (degree, 아래로 숙이면 +) =====
        self.pitch_deg = 4.5   # 원하는 각도(deg)로 바꿔서 사용
        self.pitch_rad = np.deg2rad(self.pitch_deg)
        rospy.loginfo(f"livox_bev_node: pitch_deg={self.pitch_deg:.2f}, pitch_rad={self.pitch_rad:.4f}")

    def lidar_callback(self, msg):
        points = []
        for p in pc2.read_points(msg, skip_nans=True):
            x, y, z = p[0], p[1], p[2]
            points.append([x, y, z])
        points = np.array(points)
        
        # pitch 보정
        if len(points) > 0 and abs(self.pitch_rad) > 1e-6:
            c, s = np.cos(self.pitch_rad), np.sin(self.pitch_rad)
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            x_new = x * c + z * s
            y_new = y
            z_new = -x * s + z * c
            points = np.stack([x_new, y_new, z_new], axis=1)
        
        # ROI 클리핑
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        )
        points = points[mask]

        bev = np.zeros((self.bev_height, self.bev_width), dtype=np.uint8)

        if len(points) > 0:
            u = ((-points[:, 1] - self.y_min) / (self.y_max - self.y_min) * (self.bev_width - 1)).astype(np.int32)
            v = ((self.x_max - points[:, 0]) / (self.x_max - self.x_min) * (self.bev_height - 1)).astype(np.int32)
            mask2 = (u >= 0) & (u < self.bev_width) & (v >= 0) & (v < self.bev_height)
            u = u[mask2]
            v = v[mask2]
            bev[v, u] = 255

        bev_bgr = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
        bev_msg = self.bridge.cv2_to_imgmsg(bev_bgr, encoding='bgr8')
        bev_msg.header.stamp = msg.header.stamp
        bev_msg.header.frame_id = msg.header.frame_id
        self.pub.publish(bev_msg)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    node = LivoxBevNode()
    node.spin()
