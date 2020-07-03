#!/bin/bash
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

apt-get install build-essential
apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
apt-get install wget
git clone https://github.com/opencv/opencv.git
cd /opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j7
make install
cd ../../




apt-get update

#apt-get install libpcl-dev #+ nejake dve cisla

#cmake ./rotator/CMakeLists.txt
#git clone https://github.com/martin-velas/but_velodyne_lib.git
#cd but_velodyne_lib/
#mkdir bin
#cd bin
#cmake ../
#make
