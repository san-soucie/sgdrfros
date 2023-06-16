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

BUILD_TYPE ?= "Release"

.PHONY: pull clone-ros force-clone-ros clone-workspace-packages \
update-workspace-packages rosdep-update install-rosdep-packages \
build build-merge test clean purge reformat_uncrustify reformat_black \
lint_uncrustify

src:
	mkdir -p src

pull:
	vcs pull --nested

clone-ros: src
	vcs import src --input https://raw.githubusercontent.com/ros2/ros2/$(distro)/ros2.repos

force-clone-ros: src
	vcs import src --force --input https://raw.githubusercontent.com/ros2/ros2/$(distro)/ros2.repos

clone-workspace-packages:
	vcs import src < src/ros2.repos

update-workspace-packages: 
	vcs export src > src/ros2.repos

rosdep-update:
	rosdep update

install-rosdep-packages: rosdep-update
	rosdep install --from-paths src --ignore-src -y

build:
	colcon build --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On" -Wall -Wextra -Wpedantic

test:
	if [ -f install/setup.sh ]; then . install/setup.sh; fi && colcon test && colcon test-result --verbose

clean:
	colcon build --cmake-target clean

purge:
	rm -rf build install log && py3clean .

reformat_uncrustify:
	ament_uncrustify --reformat src/

reformat_black:
	black src/

lint_uncrustify:
	ament_uncrustify src/"