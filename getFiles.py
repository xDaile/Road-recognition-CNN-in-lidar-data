#!/usr/bin/env python3

import os
import re
import parameters
import pandas as pd
import numpy as np
import torch
import createTensors

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
        keyForAccesToFile=nameOfFile
        nameOfFile=type+nameOfFile
        newList.update({keyForAccesToFile:nameOfFile})
    return newList

def getListOfGroundTruthFiles():
    trainGroundTruthNames=os.listdir(parameters.gtTrainTensors)
    trainGroundTruthNames=createListofGTwithKeys(trainGroundTruthNames,parameters.gtTrainTensors)

    testGroundTruthNames=os.listdir(parameters.gtTestTensors)
    testGroundTruthNames=createListofGTwithKeys(testGroundTruthNames,parameters.gtTestTensors)
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
    if(len(os.listdir(parameters.testTensorFolder))==0 or len(os.listdir(parameters.trainTensorFolder))==0):
        print("Tensors do not exist, run bash binToPCLscript, then ./makeRotations.py and then ./rotatedPCLToTensors")
        ##createTensors.createTensors()
        ##return loadTensorsNames()
        return -1
    else:
        print("Tensors was loaded from files")
        return loadTensorsNames()
