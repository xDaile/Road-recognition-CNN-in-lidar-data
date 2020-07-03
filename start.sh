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

apt install libpcl-dev

#FAILED HERE
#mkdir build
#cd build
#cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
#make -j7
#make install
#cd ../../




apt-get update

#apt-get install libpcl-dev #+ nejake dve cisla

#cmake ./rotator/CMakeLists.txt
#git clone https://github.com/martin-velas/but_velodyne_lib.git
#cd but_velodyne_lib/
#mkdir bin
#cd bin
#cmake ../
#make
