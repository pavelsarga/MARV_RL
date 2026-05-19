#!/bin/bash
# Run this from the root of your workspace!

echo Converting workspace from ROS 2 Jazzy to Humble..."

# 1. Clean up old build cache to prevent CMake confusion
echo "Clearing old build cache..."
rm -rf build/ install/ log/

# 2. Handle grid_map (Switch to the native Humble branch)
echo "Swapping grid_map to Humble branch..."
cd src
rm -rf grid_map
git clone -b humble https://github.com/ANYbotics/grid_map.git

# Ignore the heavy plugins we don't need for RL
touch grid_map/grid_map_octomap/COLCON_IGNORE
touch grid_map/grid_map_costmap_2d/COLCON_IGNORE
touch grid_map/grid_map_demos/COLCON_IGNORE

# 3. Brute-force C++ downgrades in your custom packages
echo "Surgically downgrading cv_bridge and rosbag2 syntax..."

# Replace cv_bridge.hpp with cv_bridge.h
find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -not -path "*/grid_map/*" -exec sed -i 's|cv_bridge/cv_bridge.hpp|cv_bridge/cv_bridge.h|g' {} +

# Replace send_timestamp with time_stamp (rosbag2 struct change)
find . -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) -not -path "*/grid_map/*" -exec sed -i 's/\.send_timestamp/\.time_stamp/g' {} +

# 4. Optional: Remove generic PyTorch from pyproject.toml so Isaac Sim's GPU PyTorch isn't overwritten
echo "Patching pyproject.toml for Isaac Sim compatibility..."
find . -name "pyproject.toml" -exec sed -i '/"torch==/d' {} +
find . -name "pyproject.toml" -exec sed -i '/"torchvision==/d' {} +

cd ..
echo "Done! You can now run 'colcon build' inside the Isaac Sim Humble container."