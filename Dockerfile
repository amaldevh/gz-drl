# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

FROM ubuntu:24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
#nvidia/cuda:12.0.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    cmake \
    git \
    wget \
    libboost-all-dev \
    libeigen3-dev \
    curl \
    nano \
    sudo \
    build-essential \
    pkg-config \
    libyaml-cpp-dev \
    gcc-11 \
    g++-11 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN . /opt/venv/bin/activate && pip install --upgrade pip
RUN . /opt/venv/bin/activate && pip install \
         numpy pytest opencv-python \
         torch torchrl pyaml optree dm_env \
          gymnasium gym stable_baselines3 \
          tqdm numpy==1.26 pybind11==2.13.1 scipy
RUN sudo apt-get update && sudo apt-get install -y lsb-release gnupg
RUN sudo curl https://packages.osrfoundation.org/gazebo.gpg\
     --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main"\
         | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
RUN sudo apt-get update
RUN sudo apt-get install gz-jetty -y
RUN cd /tmp && git clone https://github.com/google/glog.git \
    && cmake -B build -S glog && cmake --build build --target install \
    && rm -rf /tmp/glog
RUN sudo apt update -y && sudo apt install locales -y
RUN sudo locale-gen en_US en_US.UTF-8
RUN sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN sudo apt install software-properties-common -y
RUN sudo add-apt-repository universe -y
RUN curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb"
RUN sudo dpkg -i /tmp/ros2-apt-source.deb 
RUN sudo apt update -y && sudo DEBIAN_FRONTEND=noninteractive apt install ros-dev-tools -y
RUN sudo apt update -y && sudo apt-get upgrade -y
RUN sudo apt-get install ros-jazzy-desktop -y
RUN . /opt/venv/bin/activate && pip install empy catkin_pkg lark
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN sudo apt-get install libbenchmark-dev -y
WORKDIR /workspace/gzdrl
COPY . .
RUN source /opt/venv/bin/activate \
    && source /opt/ros/jazzy/setup.bash \
    && python -m pip install '.[rl,test]'
