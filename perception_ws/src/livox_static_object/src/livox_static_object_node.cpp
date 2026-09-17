#include "livox_static_object.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "livox_static_object_node");
    ros::NodeHandle nh, pnh("~");
    livox_static_object::StaticObjectDetector node(nh, pnh);
    ros::spin();
    return 0;
}
