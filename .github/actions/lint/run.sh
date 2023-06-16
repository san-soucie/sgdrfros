#!/bin/bash
set -e

make clone-workspace-packages
make install-rosdep-packages
ament_${LINTER} src/