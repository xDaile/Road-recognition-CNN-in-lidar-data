#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
        -------------------------------------------------------------------------------------------
        |    Author: Michal Zelenak                                                               |
        |    BUT Faculty of Information Technology                                                |
        |    This is code written for the bachelor thesis                                         |
        |    Project: Object Detection in the Laser Scans Using Convolutional Neural Networks     |
        -------------------------------------------------------------------------------------------
"""
import os

statsTestFolder="./stats/test"
statsTrainFolder="./stats/train"
rotatedStats="./stats/rotatedStats/"
stats=["density","maxEl","meanEL","meanRef","minEL","stdEL"]
testTensorFolder="./Dataset/test_Tensors/"
trainTensorFolder="./Dataset/trainTensors/"
modelSavedFile="./Model.tar"
rotatedPCLFiles="./pclFiles/rotatedPCL/"
gtTestTensors="./Dataset/gtTensors/test_/"
gtTrainTensors="./Dataset/gtTensors/train/"
pclFiles="./pclFiles/"
groundTruthImages="./GroundTruth/"
originalGT="./groundTruth/"
ClassForPointOutOfRotation=3
ClassForPointWhichCameraDoNotSaw=2

#grid parameters
xDownBoundary = 6
xUpBoundary = 46
yDownBoundary = -10
yUpBoundary = 10
