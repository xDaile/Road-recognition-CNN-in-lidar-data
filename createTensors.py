#!/usr/bin/env python3
import getFiles
import parameters
import os
import torch

def newTestTensorName(file):
    return parameters.testTensorFolder+file.split("/")[3][:-1]

def newTrainTensorName(file):
    return parameters.trainTensorFolder+file.split("/")[3][:-1]

#function that create new tensors in parameters.stats****Folder
def createTensors():
    shortNamesOfTestFiles=getFiles.getListOfShortenedFileNames(os.listdir(parameters.statsTestFolder),"test")
    shortNamesOfTrainFiles=getFiles.getListOfShortenedFileNames(os.listdir(parameters.statsTrainFolder),"train")
    print("Creating test tensors")
    for file in shortNamesOfTestFiles:
        nameOfFileWithFutureTensor=newTestTensorName(file)
        torch.save(getFiles.getStatsAboutFile(file),nameOfFileWithFutureTensor)
    print("Creating train tensors")
    for file in shortNamesOfTrainFiles:
        nameOfFileWithFutureTensor=newTrainTensorName(file)
        torch.save(getFiles.getStatsAboutFile(file),nameOfFileWithFutureTensor)

def main():
    createTensors()
