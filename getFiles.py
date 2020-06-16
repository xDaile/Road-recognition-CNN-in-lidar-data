#!/usr/bin/env python3

import os
import re
import parameters
import pandas as pd
import numpy as np
import torch

import cv2

    #img=transformImage(cv2.imread("./DatasetForUse/test/um_000001_gt.png"))
    #cv2.imshow("image",img)
    #cv2.waitKey(300)
    #cv2.destroyAllWindows()
    #1 - road
    #2 - not road
    #3 - not used for compute

def filterDensityName(par):
    count=len(re.findall(parameters.stats[0],par))
    if(count>0):
        return True
    return False

def filterGroundTruthFiles(fileName):
    count=len(re.findall(r'gt',fileName))
    if(count>0):
        return True
    return False

def newTestTensorName(file):
    return parameters.testTensorFolder+file.split("/")[3][:-1]

def newTrainTensorName(file):
    return parameters.trainTensorFolder+file.split("/")[3][:-1]

#not used
def transformImage(img):
    i=0
    j=0
    for row in img:
        for matrix in row:
            #road
            if (matrix[0]==1):
                img[i][j][0]=255
                img[i][j][1]=0
                img[i][j][2]=0
            #not road
            if(matrix[0]==2):
                img[i][j][0]=125
                img[i][j][1]=125
                img[i][j][2]=125
            #black collor ground truth pixels will not be used in computing, because it is not in the good area
            if(matrix[0]==3):
                img[i][j][0]=0
                img[i][j][1]=0
                img[i][j][2]=0
            j=j+1
        i=i+1
        j=0
    return img

def createTensors():
    shortNamesOfTestFiles=getListOfShortenedFileNames(os.listdir(parameters.testFolder),"test")
    shortNamesOfTrainFiles=getListOfShortenedFileNames(os.listdir(parameters.trainFolder),"train")
    print("Creating test tensors")
    for file in shortNamesOfTestFiles:
        nameOfFileWithFutureTensor=newTestTensorName(file)
        torch.save(getStatsAboutFile(file),nameOfFileWithFutureTensor)
    print("Creating train tensors")
    for file in shortNamesOfTrainFiles:
        nameOfFileWithFutureTensor=newTrainTensorName(file)
        torch.save(getStatsAboutFile(file),nameOfFileWithFutureTensor)

def getListOfShortenedFileNames(param,type):
    #get list of train and test files
    folder=""
    if(type=="test"):
        folder=parameters.testFolder
    else:
        folder=parameters.trainFolder
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
    trainGroundTruthNames=os.listdir("./groundTruthTensors/train")
    #trainGroundTruthNames=list(filter(filterGroundTruthFiles,trainGroundTruthNames))
    trainGroundTruthNames=createListofGTwithKeys(trainGroundTruthNames,"./groundTruthTensors/train")


    testGroundTruthNames=os.listdir("./groundTruthTensors/test")
    #testGroundTruthNames=list(filter(filterGroundTruthFiles,testGroundTruthNames))
    testGroundTruthNames=createListofGTwithKeys(testGroundTruthNames,"./groundTruthTensors/test")
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
    if(len(os.listdir(parameters.testTensorFolder))!=30 or len(os.listdir(parameters.trainTensorFolder))<259):
        print("Tensors will be saved and returned, next time tensors will be only loaded")
        createTensors()
        return loadTensorsNames()
    else:
        print("Tensors was loaded from files")
        return loadTensorsNames()
