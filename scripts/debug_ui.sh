colcon build --packages-select rodeo_controller --symlink-install
source install/setup.bash

export LD_LIBRARY_PATH=/opt/ros/kilted/opt/zenoh_cpp_vendor/lib:$LD_LIBRARY_PATH
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

ros2 launch rodeo_controller ui_real.launch.py
