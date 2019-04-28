#!/usr/bin/env bash

sudo apt-get update
sudo apt-get upgrade

# Install Docker
# https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce
sudo apt-get install -y curl \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# https://marmelab.com/blog/2018/03/21/using-nvidia-gpu-within-docker-container.html
# Install NVIDIA repo metadata
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg --install cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
# Install CUDA GPG key
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update

sudo apt-get install -y -o Dpkg::Options::="--force-overwrite" nvidia-418
sudo apt-get install -y -o Dpkg::Options::="--force-overwrite" --fix-broken

# Install NVidia Docker
# https://nvidia.github.io/nvidia-docker/
# https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd
sudo groupadd docker

# Postinstall - to be run as user
# https://docs.docker.com/install/linux/linux-postinstall/
#
sudo usermod -aG docker $USER
#sudo reboot now



