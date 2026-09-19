#!/usr/bin/env bash
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib
exec /isaac-sim/python.sh "$(dirname "$0")/spot_warehouse.py" "$@"
