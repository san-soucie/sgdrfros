# Copyright 2023 John San Soucie

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PKG_DIR=$(shell pwd)
SRC_DIR=$(shell dirname $(PKG_DIR))
ROS2_WS=$(shell dirname $(SRC_DIR))


BUILD_TYPE ?= "Release"

.PHONY: pull clone-ros force-clone-ros clone-workspace-packages \
update-workspace-packages rosdep-update install-rosdep-packages \
build build-merge test clean purge reformat_uncrustify reformat_black \
lint_uncrustify cd-ws docs

pull:
	vcs pull --nested

clone-ros: src
	cd ${ROS2_WS} && vcs import src --input https://raw.githubusercontent.com/ros2/ros2/$(distro)/ros2.repos

force-clone-ros: src
	cd ${ROS2_WS} && vcs import src --force --input https://raw.githubusercontent.com/ros2/ros2/$(distro)/ros2.repos

clone-workspace-packages:
	vcs import ${SRC_DIR} &&  < ${SRC_DIR}/ros2.repos

update-workspace-packages: 
	vcs export ${SRC_DIR} > ${SRC_DIR}/ros2.repos

rosdep-update:
	cd ${ROS2_WS} && rosdep update

install-rosdep-packages: rosdep-update
	cd ${ROS2_WS} && rosdep install --from-paths src --ignore-src -y

build:
	cd ${ROS2_WS} && colcon build --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On" -Wall -Wextra -Wpedantic

test:
	cd ${ROS2_WS} && colcon test && colcon test-result --verbose

clean:
	cd ${ROS2_WS} && colcon build --cmake-target clean

purge:
	cd ${ROS2_WS} && rm -rf build install log site && py3clean .

reformat_uncrustify:
	ament_uncrustify --reformat sgdrf
	ament_uncrustify --reformat sgdrf_interfaces

reformat_black:
	black sgdrf

lint_uncrustify:
	ament_uncrustify sgdrf
	ament_uncrustify sgdrf_interfaces

docs:
	cd sgdrf/docs && make html