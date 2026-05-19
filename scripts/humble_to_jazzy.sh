#!/bin/bash
# Run this from the root of your workspace!

echo "Converting workspace from ROS 2 Humble back to Jazzy..."

# 1. Clean up old build cache to prevent CMake confusion
echo "Clearing old build cache..."
rm -rf build/ install/ log/

# 2. Handle grid_map (Switch to the native Jazzy/ROS2 branch)
echo "Swapping grid_map to Jazzy branch..."
cd src
rm -rf grid_map
# Note: ANYbotics usually uses 'jazzy' or the main 'ros2' branch for modern versions
git clone -b jazzy https://github.com/ANYbotics/grid_map.git || git clone -b ros2 https://github.com/ANYbotics/grid_map.git

# Ignore plugins if you still don't want them natively
touch grid_map/grid_map_octomap/COLCON_IGNORE
touch grid_map/grid_map_costmap_2d/COLCON_IGNORE
touch grid_map/grid_map_demos/COLCON_IGNORE

# 3. Brute-force C++ upgrades in your custom packages
echo "Surgically upgrading cv_bridge and rosbag2 syntax..."

# Replace cv_bridge.h with cv_bridge.hpp
find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -not -path "*/grid_map/*" -exec sed -i 's|cv_bridge/cv_bridge.h|cv_bridge/cv_bridge.hpp|g' {} +

# Replace time_stamp with send_timestamp (rosbag2 struct change)
find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -not -path "*/grid_map/*" -exec sed -i 's/\.time_stamp/\.send_timestamp/g' {} +

cd ..
echo "✅ Done! You can now run 'colcon build' natively on your Ubuntu 24.04 desktop."