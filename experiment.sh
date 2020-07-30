#!/bin/bash
# -*- coding: utf-8 -*-
#   --------------------------------------------------------------------------------------------
#   |    Author: Michal Zelenak                                                                  |
#   |    BUT Faculty of Information Technology                                                   |
#   |    This is code written for the bachelor thesis                                            |
#   |    Project: Object Detection in the Laser Scans Using Convolutional Neural Networks        |
#   -----------------------------------------------------------------------------------------


#if pclFiles directory do not exists - it is first run
if [ -d "./pclFiles" ]
then
  echo "pcl files will be used from directory ./pclFiles, if you want to generate them again run \$ bash binToPCL.sh"
else
  mkdir pclFiles
  echo "pclFiles directory created"
  bash binToPCL.sh
fi

#check if the ground truth images were generated into file ./GroundTruth
if [ -d "./GroundTruth" ]
then
  echo "GroundTruth images in ./GroundTruth directory will be used, if you want to generate them again run ./GTpngToNP.py"
else
  mkdir GroundTruth
  echo "Ground Truth directory created"
  ./GTpngToNP.py
fi

#creater rotated versions with projected GT
./createRotatedPcl.py

#extract input tensors and ground truth from transformated point clouds
./createNetworkInputAndGT.py

#startTraining
./baseCNN.py
