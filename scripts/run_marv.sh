colcon build --allow-overriding grid_map_core grid_map_cv grid_map_filters grid_map_loader grid_map_msgs grid_map_ros grid_map_rviz_plugin grid_map_visualization
source ~/workspaces/robot_rodeo_gym_ws/install/setup.bash

export LD_LIBRARY_PATH=/opt/ros/kilted/opt/zenoh_cpp_vendor/lib:$LD_LIBRARY_PATH
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

if ! pgrep -f "rmw_zenohd" > /dev/null; then
    ros2 run rmw_zenoh_cpp rmw_zenohd &
    echo "Started Zenoh router (PID $!)"
    sleep 1
fi

ros2 launch robot_rodeo_gym gazebo_marv.launch.py teleop:=true "$@"

