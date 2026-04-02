# Source ROS2 Environment

**execute the command below when starting a terminal**

```
source /opt/ros/humble/setup.bash

```

**or: add to \~/.bashrc and source**

1. open \~/.bashrc in any way

2. add "source /opt/ros/humble/setup.bash" to the end

3. source \~/.bashrc

# Create a workspace and a package

reference: <https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html>
<https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html>

1. create a workspace

```
mkdir -p ~/exp1_ws/src
cd ~/exp1_ws/src

```

1. create a package "visualize"

```
ros2 pkg create --build-type ament_cmake visualize

```

# Download launch file and robot description

1. download related files from:

2. the workspace should look like:

```
exp1_ws/
    src/
        visualize/
            include/
            launch/
                rviz_vis.launch.py
            robot_description/
                ur3/
                    meshes/
                    ur3.xacro
            scripts/
                rviz_visualize.py
            CMakeLists.txt
            package.xml

```

1. edit CMakeLists.txt, add following lines

```
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY launch robot_description
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/rviz_visualize.py
  DESTINATION lib/${PROJECT_NAME}
)

```

1. install dependencies

```
sudo apt install python3-colcon-common-extensions   
sudo apt install ros-humble-xacro
sudo apt install ros-humble-joint-state-publisher

```

1. build the workspace and source

```
cd ~/exp1_ws
colcon build
source install/setup.bash

```

# Task1: Visualize UR3 URDF model

1. launch rviz2

```
ros2 launch visualize rviz_vis.launch.py

```

1. add topics in rviz2

* GlobalOptions - Fixed Frame: /visualize/base\_link

* Add - By display type: TF

* Add - By display type: RobotModel

* RobotModel - Description Topic: /visualize/robot\_description

* TF Prefix: visualize

1. visualize tf tree

```
sudo apt-get install ros-humble-rqt-tf-tree
ros2 run rqt_tf_tree rqt_tf_tree --force-discover

```

**Now you can see a virtual robot with corresponding frames in rivz2**

# Task2: Publish JointState message and move the robot

1. edit scripts/rviz\_visualize.py (init and publish\_robot\_joint\_states function)

   Hint: you should write a publisher to send joint state message (sensor\_msgs.msg.JointState). Note that here you should know the joint names in urdf model to send corresponding joint positions. For ur3, it has 6 joints and you can refer to ur3.xacro to find the joint names.

2. build the workspace

```
cd ~/exp1_ws
colcon build
source install/setup.bash

```

1. run rviz\_visualize.py

```
ros2 run visualize rviz_visualize.py

```

# Task3: Lookup TF2 transform and publish a Marker

1. edit scripts/rviz\_visualize.py (init and publish\_marker function)

   Hint: you should write a tf buffer and listener in init, use lookupTransform to get the transformation from wrist\_3\_link to base\_link. define the position/scale/color of the marker (visualization\_msgs.msg.Marker) and publish the message.

2. build the workspace

```
cd ~/exp1_ws
colcon build

```

1. add the Marker topic in rviz and run rviz\_visualize.py

```
ros2 run visualize rviz_visualize.py

```

​
