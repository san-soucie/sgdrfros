# from: https://github.com/athackst/dockerfiles

###########################################
# Base image 
###########################################
FROM osrf/ros:iron-desktop-full-jammy as dev

LABEL org.opencontainers.image.source="https://github.com/san-soucie/sgdrfros"

ENV APP_NAME=sgdrfros
ENV ROS_DISTRO=IRON
ENV DEBIAN_FRONTEND=noninteractive
ENV AMENT_PREFIX_PATH=/opt/ros/iron
ENV COLCON_PREFIX_PATH=/opt/ros/iron
ENV LD_LIBRARY_PATH=/opt/ros/iron/lib
ENV PATH=/opt/ros/iron/bin:$PATH
ENV PYTHONPATH=/opt/ros/iron/lib/python3.10/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2
ENV LANG en_US.UTF-8
ENV AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1

# Install language
# Install timezone
# Update packages
# Install common programs
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \ 
  && apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
  tzdata \
  locales \
  curl \
  gnupg2 \
  lsb-release \
  sudo \
  software-properties-common \
  wget \
  bash-completion \
  build-essential \
  cmake \
  gdb \
  git \
  git-core \
  sudo \
  openssh-client \
  python3-argcomplete \
  python3-pip \
  ros-dev-tools \
  vim \
  python3-sphinx \
  tini \
  cmake \
  git \
  python3-setuptools  \
  python3-bloom  \
  python3-colcon-common-extensions  \
  python3-rosdep  \
  python3-vcstool  \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

# Initialize rosdep, if required
RUN rosdep init || echo "rosdep already initialized"

# Create a non-root user
ARG USERNAME=ros
ARG USER_UID=10000
ARG USER_GID=10001
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up autocompletion for user
RUN echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc \
  && echo "if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash; fi" >> /home/$USERNAME/.bashrc

RUN python3 -m pip install sphinx-rtd-theme bump2version

ENV DEBIAN_FRONTEND=

ARG WORKSPACE=/workspaces
RUN mkdir -p ${WORKSPACE}/src
WORKDIR ${WORKSPACE}
RUN bash -c 'echo "yaml https://raw.githubusercontent.com/san-soucie/rosdistro/python-pyro-ppl-pip/rosdep/python.yaml" > /etc/ros/rosdep/sources.list.d/10-python-pyro-ppl-pip.list'
RUN echo $'#!/bin/bash \n\
  set -e \n\
  # setup ros2 environment \n\
  source "/opt/ros/iron/setup.bash" -- \n\
  exec "\$@" ' > /entrypoint.sh
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/ros/.bashrc
ENTRYPOINT [ "/tini" "--" "/entrypoint.sh" ]

FROM dev as build

ARG WORKSPACE=/workspaces
ARG BUILD_TYPE=release

COPY . ${WORKSPACE}/src/${APP_NAME}
RUN rosdep update && rosdep install --from-paths src --ignore-src -y
RUN colcon build --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On" -Wall -Wextra -Wpedantic

FROM dev

COPY --from=build ${WORKSPACE}/install ${WORKSPACE}/install
RUN echo $'#!/bin/bash \n\
  set -e \n\
  # setup ros2 environment \n\
  source "/opt/ros/iron/setup.bash" -- \n\
  source "${WORKSPACE}/install/local_setup.bash" -- \n\
  exec "\$@" ' > /entrypoint.sh

