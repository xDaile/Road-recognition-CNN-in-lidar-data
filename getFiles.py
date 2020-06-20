#!/usr/bin/env python3

import os
import re
import parameters
import pandas as pd
import numpy as np
import torch
import createTensors
import cv2

    #1 - road
    #2 - not road
    #3 - not used for compute


def filterDensityName(par):
    count=len(re.findall(parameters.stats[0],par))
    if(count>0):
        return True
    return False


def getListOfShortenedFileNames(param,type):
    #get list of train and test files
    folder=""
    if(type=="test"):
        folder=parameters.statsTestFolder
    else:
        folder=parameters.statsTrainFolder
    files=list(filter(filterDensityName,param))
    i=0
    newList=[]
    while(i<len(files)):
        match=re.split(r'[_]', files[i])
        match=folder+"/"+match[0]+"_"+match[1]+"_"
        newList.append(match)
        i=i+1
    return newList

def getStatsAboutFile(shortenedNameOfFile):
    #get stats about one file
    statsAboutFile=[]
    for typeOfStatistics in parameters.stats:
        fullNameOfFile=shortenedNameOfFile+typeOfStatistics+".csv"
        dataOriginal=pd.read_csv(fullNameOfFile, index_col =False,header = None).astype('float')
        torch_tensor=torch.tensor(dataOriginal.values)
        torch_tensor=torch_tensor.float()
        statsAboutFile.append(torch_tensor)
    return torch.stack(statsAboutFile)

def createListofGTwithKeys(list,type):
    newList={}
    for nameOfFile in list:
        keyForAccesToFile=nameOfFile[:-3]
        nameOfFile=type+"/"+nameOfFile
        newList.update({keyForAccesToFile:nameOfFile})
    return newList

def getListOfGroundTruthFiles():
    trainGroundTruthNames=os.listdir(parameters.groundTruthTrainTensorsFolder)
    trainGroundTruthNames=createListofGTwithKeys(trainGroundTruthNames,parameters.groundTruthTrainTensorsFolder)

    testGroundTruthNames=os.listdir(parameters.groundTruthTestTensorsFolder)
    testGroundTruthNames=createListofGTwithKeys(testGroundTruthNames,parameters.groundTruthTestTensorsFolder)
    return {"test":testGroundTruthNames, "train":trainGroundTruthNames}

def fullPathAndKeyForTensor(list,place):
    newList={}
    for key in list:
        fullName=place+key
        newList.update({key:fullName})
    return newList

def loadTensorsNames():
    testTensorsList=os.listdir(parameters.testTensorFolder)
    testTensorsList=fullPathAndKeyForTensor(testTensorsList,parameters.testTensorFolder)
    trainTensorsList=os.listdir(parameters.trainTensorFolder)
    trainTensorsList=fullPathAndKeyForTensor(trainTensorsList,parameters.trainTensorFolder)
    return {"test":testTensorsList, "train":trainTensorsList}

def getListOfIDs():
    testList=os.listdir(parameters.testTensorFolder)
    trainList=os.listdir(parameters.trainTensorFolder)
    return {"test":testList, "train":trainList}

def loadListOfTensors():
    #check if there are some tensor files, if not use loadStats for sure function
    if(len(os.listdir(parameters.testTensorFolder))!=120 or len(os.listdir(parameters.trainTensorFolder))<1036):
        print("Tensors will be saved and returned, next time tensors will be only loaded")
        createTensors.createTensors()
        return loadTensorsNames()
    else:
        print("Tensors was loaded from files")
        return loadTensorsNames()
