cmake_minimum_required(VERSION 3.0.2)
project(livox_static_object)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  geometry_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# catkin_install_python(PROGRAMS
#   scripts/lidar_clustering.py
#   scripts/euclidean_clustering.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
