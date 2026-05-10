# 2025 제4회 국제 대학생 EV자율주행 경진대회 — Perception 모듈

*Advanced 자율주행 모빌리티 레이스1/2 부문 최우수상*

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![ROS](https://img.shields.io/badge/-ROS-22314E?style=for-the-badge&logo=ROS)

본 레포는 Clothoid-R 자율주행 시스템의 **perception 모듈**입니다. 카메라/라이다 기반 객체 검출과 센서 퓨전 파이프라인을 포함합니다.

## 팀원

<table>
<tr>
<td align="center"><a href="https://github.com/JaGyeong1024"><img src="https://avatars.githubusercontent.com/u/105336903?v=4" width="100px;" alt=""/><br /><sub><b>구자경</b></sub></a><br />Perception Architecture, LiDAR Clustering</td>
<td align="center"><a href="https://github.com/gyeongseoMin"><img src="https://avatars.githubusercontent.com/u/105336903?v=4" width="100px;" alt=""/><br /><sub><b>서준혁</b></sub></a><br />DL based LiDAR </td>
<td align="center"><a href="https://github.com/SeoooooNyeong"><img src="https://avatars.githubusercontent.com/u/105336903?v=4" width="100px;" alt=""/><br /><sub><b>김민재</b></sub></a><br />Camera, DL Pruning</td>
<td align="center"><a href="https://github.com/JOONHOGITHUB"><img src="https://avatars.githubusercontent.com/u/105336903?v=4" width="100px;" alt=""/><br /><sub><b>지연수</b></sub></a><br />Camera, DL</td>
<td align="center"><a href="https://github.com/leeharam2004"><img src="https://avatars.githubusercontent.com/u/105336903?v=4" width="100px;" alt=""/><br /><sub><b>유인석</b></sub></a><br />Camera, H/W</td>
</tr>
</table>

## 시스템 구성

세 개의 perception 파이프라인이 병렬로 동작하며, 결과는 planning(local_path) 노드로 전달됩니다.

| 파이프라인 | 입력 | 출력 토픽 | 설명 |
|---|---|---|---|
| Livox 유클리디안 클러스터링 | `/livox/lidar` | `/jagyeong` | 거리 가중치 유클리디안 클러스터링 + 칼만 추적 |
| Livox + 카메라 YOLO 센서 퓨전 | `/livox/lidar`, `/camera/image_raw/compressed`, `/yolov8_pub` | `/minjae` | message_filters 동기화 후 LiDAR 포인트를 카메라 좌표로 투영 → bbox 매칭 |
| 벨로다인 BEV YOLO | `/velodyne` | `/junhyeok` | BEV 변환 후 YOLO 검출 + OC-SORT 추적 |

### Livox + 카메라 퓨전 파이프라인

```
ROS Topic 동기화 (LiDAR + 카메라 + YOLO)
        ↓
LiDAR 포인트 → 카메라 좌표계 투영
        ↓
YOLO bbox와 매칭된 LiDAR 포인트 필터링
        ↓
ROI 추출 및 RANSAC 기반 Ground 제거
        ↓
Point Cloud 군집화 및 중심점(Centroid) 계산
        ↓
Kalman Filter 기반 객체 추적
        ↓
결과 시각화 및 PointCloud 발행 (/minjae)
```

## 패키지 구성

| 패키지 | 역할 |
|---|---|
| `Object_detect/` | Livox+카메라 퓨전 노드(C++), Livox 유클리디안 클러스터링(Python), 벨로다인 BEV 검출(Python) |
| `yolov8/` | 카메라 YOLO 검출 노드 (`/yolov8_pub` 발행) |
| `lidar_object_detec/` | 벨로다인 BEV 모델 가중치 보관 |
| `detect_msgs/` | `Yolo_Objects` 등 메시지 정의 |
| `yolov8_prune/` | 커스터마이즈된 ultralytics 라이브러리 (editable install) |

## Requirements

- Ubuntu 20.04 / ROS Noetic
- Conda

## Installation

기존 ultralytics가 설치돼 있다면 먼저 제거합니다.

```bash
pip uninstall ultralytics
```

워크스페이스 클론 및 환경 세팅:

```bash
git clone https://github.com/JaGyeong1024/Perception_Clothoid-R_JEJU.git
cd Perception_Clothoid-R_JEJU

conda env create -f environment.yml
conda activate clothoid

cd yolov8_prune
pip install -e .
pip install torch torchvision pyyaml

source /opt/ros/noetic/setup.bash
```

## Build

```bash
cd Perception_Clothoid-R_JEJU
catkin_make
source devel/setup.bash
```

## Run

세 파이프라인을 모두 띄우려면 **터미널 4개**가 필요합니다 (roscore + YOLO + 퓨전/BEV + 유클리디안).

각 터미널에서 환경 source:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
conda activate clothoid
```

### 터미널 1 — roscore

```bash
roscore
```

### 터미널 2 — 카메라 YOLO

`/yolov8_pub` 토픽이 먼저 발행돼야 퓨전 노드가 동작하므로 가장 먼저 띄웁니다.

```bash
roslaunch yolov8 yolo_detect.launch
```

### 터미널 3 — Livox+카메라 퓨전 + 벨로다인 BEV

`detect.launch` 하나에 두 노드가 묶여 있습니다.

```bash
roslaunch Object_detect detect.launch
```

### 터미널 4 — Livox 유클리디안 클러스터링

```bash
rosrun Object_detect euclidean_clustering.py
```

## 알려진 이슈

- **벨로다인 BEV 출력 토픽 미스매치**: `bev_object_detector.py`는 현재 `/detected_objects`로 발행하지만, planning(`local_path_real.cpp`)은 `/junhyeok`을 구독합니다. `bev_object_detector.py:57` 또는 launch에서 remap으로 통일 필요.
- **절대 경로 하드코딩**: `Object_detect/launch/detect.launch`의 `model_path`, `bev_object_detector.py:7`의 `MODEL_PATH`가 빌드 머신별 절대 경로로 설정돼 있어 환경에 맞게 수정 필요.
