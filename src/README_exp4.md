# Install Dependencies
1. install realsense ros driver
```
sudo apt install ros-humble-librealsense2*
sudo apt install ros-humble-realsense2-*
```

2. install open3d
```
pip install open3d
```

3. add this line into ~/.bashrc and source it
```
export XDG_SESSION_TYPE=x11
```

# Launch Realsense node and visualize in rivz2 

1. launch realsense
```
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true
```
2. check the topics (/camera/camera/xxx are the realsense realted topics)
```
ros2 topic list
```
3. visualize in rviz2 (first launch rivz2)
Add - By topic - camera/camera/color/image_raw/Image
Add - By topic - camera/camera/depth/color/points/PointCloud2


# Create a package and a python file
1. create a new package 'detection' (refer to exp1/exp2)
2. create a python file 'src/detection/scripts/detect_cube.py' 
3. modify 'src/detection/CMakeLists.txt'


# Task1: Subscribe the RGB and Depth topic
1. write a CubeDetector node(refer to exp1)
2. create subscriber for rgb topic '/camera/camera/color/image_raw', depth topic '/camera/camera/aligned_depth_to_color/image_raw', camerainfo topic '/camera/camera/color/camera_info'
3. use message_filters.ApproximateTimeSynchronizer to synchronize the rgb and depth topic
4. write the callback function for image subscriber and camerainfo subscriber
5. use CvBridge() to transfer received image message to numpy
6. use opencv to visualize rgb image

# Task2: Segment the cube and detect the contours
1. transfer the rgb image to hsv space with cv2.cvtColor()
2. set hsv thresholds and segment the region with cv2.inRange()
3. find contours from masks with cv2.findContours()
4. use cv2.approxPolyDP() to check which contours have 4 vertexes
5. choose the contour with max area, calculate its center point

# Task3: Project 2D points to 3D space
1. get 2D positions in the chosen contour
2. use the K matrix in camerainfo topic, project 3D points to 3D space
3. use open3d to visualize

# Task4: Use ICP to get the transformation information
1. set a virtual point cloud
2. use o3d.pipelines.registration.registration_icp() to get the transformation between virtual point cloud and real point cloud

# Task5: Send the TF transform
1. referred to: https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
2. define a TransformBroadcaster and send this transform (header.frame_id:'camera_color_optical_frame', child_frame_id:'cube')
3. add TF in rviz2 to visualize 'cube'