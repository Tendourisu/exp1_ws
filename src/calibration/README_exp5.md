# Task 1: hand eye calibration

## Clone easy_handeye2 and aruco_ros

1. clone aruco_ros (if download speed is too low, extract from the zip file in the provided documents)


```
cd exp1_ws/src
mkdir calibration
cd calibration
git clone https://github.com/pal-robotics/aruco_ros.git -b humble-devel

```

2. clone easy_handeye2 (if download speed is too low, extract from the zip file in the provided documents)


```
git clone https://github.com/THU-DA-Robotics/easy_handeye2.git

```

3. install dependencies


```
cd exp1_ws
rosdep install --from-paths src --ignore-src --rosdistro humble -y

```

4. build


```
colcon build
source install/setup.bash

```

## Hand Eye Calibration

1. attach the aruco marker to the wrist3 link
2. launch realsense camera


```
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true

```

3. launch ur robot driver


```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 robot_ip:=192.168.56.3

```

4. change the "camera_frame" to "camera_color_optical_frame", and launch the calibration node


```
ros2 launch easy_handeye2 realsense_ur_calib.launch.py 

```

5. view the aruco results


```
ros2 run rqt_image_view rqt_image_view 

```

choose topic: /aruco_single/result

6. move ur3 and calibrate

## Use calibration result

1. the calibration result file is at /home/xxx/.ros2/easy_handeye2/calibrations/xxx, you need to modify publish.launch.py to load your result file.
2. launch realsense camera


```
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true

```

3. launch ur robot driver


```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 robot_ip:=192.168.56.3

```

4. launch publish.launch.py


```
ros2 launch easy_handeye2 publish.launch.py

```

1. in rviz, check the tf of camera_link is published correctly or not

# Task 2: your first suction attempt!

Hint 1: you can create a new file named `detect_cube_and_suck.py` under `detection/scripts/`.

Hint 2: for the control part, simply use moveL since there is no obstacles.

Hint 3: the code for suction

First, change the permission of the device with "sudo chmod 777 /dev/ttyUSB0"

```
import serial

serial_suction = serial.Serial('/dev/ttyUSB0',115200,timeout=1,bytesize=8)
serial_suction.write(bytes.fromhex('01 06 00 02 00 01 E9 CA')) # suck
serial_suction.write(bytes.fromhex('01 06 00 02 00 02 A9 CB')) # release

```


