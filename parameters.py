#!/usr/bin/env python3
import os

statsTestFolder="./stats/test"
statsTrainFolder="./stats/train"
stats=["density","maxEl","meanEL","meanRef","minEL","stdEL"]
testTensorFolder="./tensors/test/"
trainTensorFolder="./tensors/train/"
groundTruthTestFilesFolder="./groundTruth/test/"
groundTruthTrainFilesFolder="./groundTruth/train/"
groundTruthTestTensorsFolder="./groundTruthTensors/test/"
groundTruthTrainTensorsFolder="./groundTruthTensors/train/"
modelSavedFile="./model.tar"
