#!/bin/bash

#instalation of the packages which are needed for this project

apt-get update
apt-get install libeigen3-dev
pip install pandas
pip install matplotlib
pip install torch
pip install numpy
apt install screen
pip install notify-run
notify-run configure https://notify.run/2sgVnBxNtkkPi2oc
apt install cmake

apt install libpcl-dev

#build pclRotator
cd pclRotator
cmake ./CMakeLists.txt
make
cd
cd ./Road-recognition-CNN-in-lidar-data

#Create directory structure
mkdir ./Dataset
mkdir ./Dataset/gtTensors
mkdir ./Dataset/gtTensors/test_
mkdir ./Dataset/gtTensors/train
mkdir ./Dataset/test_Tensors
mkdir ./Dataset/trainTensors
mkdir ./pclFiles/pclFilesWithClasses
mkdir ./pclFiles/rotatedPCL



apt-get update
git config --global credential.helper store
