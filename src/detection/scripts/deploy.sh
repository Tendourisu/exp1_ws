# ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true depth_module.depth_profile:=1280x720x30 rgb_camera.color_profile:=1280x720x30

ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 robot_ip:=192.168.56.3

ros2 launch easy_handeye2 publish.launch.py

python3 src/detection/scripts/detect_cube_and_suck.py
