#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

/* ★ 추가 헤더 두 줄 */
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tf_map_livox_broadcaster");
    ros::NodeHandle pnh("~");

    /* 파라미터 (launch 에서 전달) */
    double utm_x   = 0.0;
    double utm_y   = 0.0;
    double yaw_deg = 0.0;
    pnh.getParam("utm_offset_x", utm_x);
    pnh.getParam("utm_offset_y", utm_y);
    pnh.getParam("yaw_deg",      yaw_deg);

    tf2_ros::TransformBroadcaster br;
    ros::Rate loop(10);   // 10 Hz

    while (ros::ok())
    {
        geometry_msgs::TransformStamped t;
        t.header.stamp = ros::Time::now();
        t.header.frame_id    = "map";
        t.child_frame_id     = "livox_frame";

        t.transform.translation.x = utm_x;
        t.transform.translation.y = utm_y;
        t.transform.translation.z = 0.0;

        double yaw_rad = yaw_deg * M_PI / 180.0;
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw_rad);
        t.transform.rotation = tf2::toMsg(q);

        br.sendTransform(t);
        loop.sleep();
    }
    return 0;
}
