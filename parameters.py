#!/usr/bin/env python3
import os

statsTestFolder="./stats/test"
statsTrainFolder="./stats/train"
rotatedStats="./stats/rotatedStats/"
stats=["density","maxEl","meanEL","meanRef","minEL","stdEL"]
testTensorFolder="./Dataset/test_Tensors/"
trainTensorFolder="./Dataset/trainTensors/"
#groundTruthTestFilesFolder="./groundTruth/test/"
#groundTruthTrainFilesFolder="./groundTruth/train/"
#groundTruthTestTensorsFolder="./gtTensors/test/"
#groundTruthTrainTensorsFolder="./gtTensors/train/"
modelSavedFile="./Model.tar"
#dirWithRotatedPCLs="./dataForRotations/pclFiles/pclFilesWithClasses/newFiles/"
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
