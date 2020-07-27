#!/bin/bash
#if pclFiles are not in the dir but bin files yes (in directory ./binFiles) run first  $ bash binToPCL.sh

#creater rotated versions with projected GT
./createRotatedPcl.py

#extract input tensors and ground truth from transformated point clouds
./createNetworkInputAndGT.py

#startTraining
./baseCNN.py
