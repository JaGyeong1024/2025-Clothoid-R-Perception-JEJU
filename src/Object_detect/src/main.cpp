#include "detection.h" // ✅ 헤더는 쌍따옴표로

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Object_Detection");
    ros::NodeHandle nodeHandle("~");
    Object_Detection detector(&nodeHandle); // ✅ 변수명 중복 없음
    ros::spin();
    return 0;
}
