#!/usr/bin/env python3
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
ClassForPointOutOfRotation=3
ClassForPointWhichCameraDoNotSaw=2

#grid parameters
xDownBoundary = 6
xUpBoundary = 46
yDownBoundary = -10
yUpBoundary = 10
