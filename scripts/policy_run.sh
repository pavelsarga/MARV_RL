source ~/workspaces/robot_rodeo_gym_ws/install/setup.bash

export LD_LIBRARY_PATH=/opt/ros/kilted/opt/zenoh_cpp_vendor/lib:$LD_LIBRARY_PATH
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ros2 launch flipper_training flipper_policy.launch.py "$@"