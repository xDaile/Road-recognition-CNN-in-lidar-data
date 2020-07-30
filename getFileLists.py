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
import re
import parameters
import pandas as pd
import numpy as np
import torch

#Classes
    #1 - road
    #2 - not road
    #3 - place where lidar was but photo not

#function for creating ground truth dict
def createDictOfGroundTruth(list,place):
    newList={}
    for nameOfFile in list:
        keyForAccesToFile=nameOfFile
        nameOfFile=place+nameOfFile
        newList.update({keyForAccesToFile:nameOfFile})
    return newList

#function for creating combination of train and test ground truth dictionaries
def getDictOfGroundTruthFiles():
    trainGroundTruthNames=os.listdir(parameters.gtTrainTensors)
    trainGroundTruthNames=createDictOfGroundTruth(trainGroundTruthNames,parameters.gtTrainTensors)

    testGroundTruthNames=os.listdir(parameters.gtTestTensors)
    testGroundTruthNames=createDictOfGroundTruth(testGroundTruthNames,parameters.gtTestTensors)
    return {"test":testGroundTruthNames, "train":trainGroundTruthNames}

#returns tensor with properties for the dicionary used in dataset
def getFullPathAndKeyForTensor(list,place):
    newList={}
    for key in list:
        fullName=place+key
        newList.update({key:fullName})
    return newList

#function for loading input tensors
def getInputTensorsNames():
    testTensorsList=os.listdir(parameters.testTensorFolder)
    testTensorsList=getFullPathAndKeyForTensor(testTensorsList,parameters.testTensorFolder)
    trainTensorsList=os.listdir(parameters.trainTensorFolder)
    trainTensorsList=getFullPathAndKeyForTensor(trainTensorsList,parameters.trainTensorFolder)
    return {"test":testTensorsList, "train":trainTensorsList}

def getListOfIDs():
    testList=os.listdir(parameters.testTensorFolder)
    trainList=os.listdir(parameters.trainTensorFolder)
    return {"test":testList, "train":trainList}

def loadListOfTensors():
    if(len(os.listdir(parameters.testTensorFolder))==0 or len(os.listdir(parameters.trainTensorFolder))==0):
        print("Tensors do not exist, run bash binToPCLscript, then ./makeRotations.py and then ./rotatedPCLToTensors")
        return -1
    else:
        print("Tensors was loaded from files")
        return getInputTensorsNames()
