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
from __future__ import print_function
from notify_run import Notify
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import numpy
import torch
import getFileLists
import parameters
import Model
import accuracyCalc
from pathlib import *
import re
import toolsForNetwork
import math

modelName="./Model.tar"
numOfClasses=3


#class for handling the point cloud file
class inputForModel():
    def __init__(self,nameOfPCLfile):
        super(inputForModel,self).__init__()
        #print("Started work with",nameOfPCLfile)
        self.loadFile(nameOfPCLfile)
        self.sortPointsInArea()
        self.pointsSortedInGrid= [[[] for i in range(200)] for j in range(400)]
        self.sortPointsInSelectedAreaIntoGrid()
        self.stats=[]
        self.countStats()
        self.tensorForModel=[]
        self.createTensor()

    def loadFile(self,nameOfPCLfile):
        try:
            self.rawFile = open(nameOfPCLfile, "r").readlines()
        except:
            print("file with input for model not found")
            exit(1)
        self.rawPoints=self.rawFile[11:]

    #sort points in the point cloud, to points which are in the area of interest, and points which are not in the area of interest
    def sortPointsInArea(self):
        i=0
        self.pointsInSelectedArea=[]
        self.unusedPoints=[]
        while(i<len(self.rawPoints)):
            # split items by every whitespace, one point contains 4 stats x,y,z,intensity
            OnePointArray = re.split(r"\s+", self.rawPoints[i])

            # last item in list is only whitespace for newline
            OnePointArray = OnePointArray[:6]
            # check if the point is in the area 400x200 metres that we will consider for counting stats
            if (
                float(OnePointArray[0]) > parameters.xDownBoundary
                and float(OnePointArray[0]) < parameters.xUpBoundary
                and float(OnePointArray[1]) > parameters.yDownBoundary
                and float(OnePointArray[1]) < parameters.yUpBoundary
            ):
                self.pointsInSelectedArea.append(OnePointArray)
            else:
                self.unusedPoints.append(OnePointArray)
            i = i + 1

    #sorting points into 2d array  - grid
    def sortPointsInSelectedAreaIntoGrid(self):
        i=0
        while i < len(self.pointsInSelectedArea):
            xCoord =int(399- int(((float(self.pointsInSelectedArea[i][0]))-6)*10))
            yCoord =int(199- int(((float(self.pointsInSelectedArea[i][1]))+10) * 10))
            self.pointsSortedInGrid[xCoord][yCoord].append(self.pointsInSelectedArea[i])
            i = i + 1

    #function that counts stats about each cell in the 2d array - grid
    def countStats(self):
        density = [[0 for i in range(200)] for j in range(400)]
        minEL = [[0 for i in range(200)] for j in range(400)]
        maxEL = [[0 for i in range(200)] for j in range(400)]
        avgELList = [[0 for i in range(200)] for j in range(400)]
        avgRefList = [[0 for i in range(200)] for j in range(400)]
        stdEL = [[0 for i in range(200)] for j in range(400)]
        i = 0
        while(i<400):
            j=0
            while(j<200):
                cellInGrid=self.pointsSortedInGrid[i][j]
                if(len(cellInGrid)==0):
                    minEL[i][j] = 0.0
                    maxEL[i][j] = 0.0
                    avgELList[i][j] = 0.0
                    avgRefList[i][j] = 0.0
                    density[i][j] = 0.0
                    stdEL[i][j] = 0.0
                else:
                    # setting the minimum and maximum with the first point in the cell(because we want to have something to compare to)
                    minEL[i][j]=float(cellInGrid[0][2])
                    maxEL[i][j]=float(cellInGrid[0][2])
                    numOfPointsInCell=len(cellInGrid)
                    density[i][j]=numOfPointsInCell
                    # average elevation
                    avgELSum=0.0
                    # average reflectivity
                    avgRefSum=0.0
                    # standard deviation of elevation(variable for computing)
                    std = 0.0
                    # going throught the list of original points in the cell

                    for t in cellInGrid:
                        # summary of elevations(z coordinate)
                        avgELSum = avgELSum + float(t[2])

                        # summary of reflectivity(r coordinate)
                        avgRefSum = avgRefSum + float(t[3])

                        # actual elevation
                        elevation = float(t[2])
                        # finding lowest point
                        if elevation < minEL[i][j]:
                            minEL[i][j] = elevation
                        # finding highest point
                        if elevation > maxEL[i][j]:
                            maxEL[i][j] = elevation

                    # counting the mean of elevation and reflectivity
                    avgEL = avgELSum / numOfPointsInCell
                    avgRef = avgRefSum / numOfPointsInCell

                    # againg going throught the points in cell
                    # for counting the standard deviation(in previous step we did not have mean - we have to count mean first and then standard deviation)
                    for t in cellInGrid:
                        std = std + ((float(t[2]) - avgELSum) * (float(t[2]) - avgELSum))
                    std = std / numOfPointsInCell
                    std = math.sqrt(std)

                    # saving the data
                    stdEL[i][j] = std
                    avgELList[i][j] = avgEL
                    avgRefList[i][j] = avgRef
                    #added points are better distributed -> count gt from those points
                j = j + 1
            i+=1
        self.stats=[density,maxEL,avgELList,avgRefList,minEL,stdEL]

    def createTensor(self):
        self.tensorForModel=torch.stack([torch.tensor(self.stats)])

#class for working with trained model
class modelWorker():
    def __init__(self,modelFileName):
        super(modelWorker,self).__init__()
        self.modelStructure=Model.Net()
        self.loadStateIntoStructure(modelFileName)

    #will load saved state from file
    def loadStateIntoStructure(self,modelFileName):
        self.savedModel=torch.load(modelFileName)
        self.modelStructure.load_state_dict(self.savedModel['model_state_dict'])
        self.modelStructure.eval()

    def useModel(self, inputForModel):
        self.tensorOutput=self.modelStructure(inputForModel)
        return self.tensorOutput

    def getTensorOutputFromModel(self,inputForModel):
        return self.useModel(inputForModel)

    def getNumpyOutputFromModel(self,inputForModel):
        tensorOutput=self.getTensorOutputFromModel(inputForModel)
        numpyAr=tensorOutput.detach().numpy()
        numpyAr=numpyAr[0]
        out = [[0 for i in range(200)] for j in range(400)]
        i=0
        while(i<400):
            j=0
            while(j<200):
                #if(i>300):
                if(numpyAr[0][i][j]>numpyAr[1][i][j] and numpyAr[0][i][j]>numpyAr[2][i][j]) :
                    out[i][j]=0#,numpyAr[2][i][j])
                else:
                    if(numpyAr[1][i][j]>numpyAr[2][i][j]):
                        out[i][j]=1
                    else:
                        out[i][j]=2
                j=j+1
            i=i+1
        return out

    def showResultFromNetwork(self,inputForModel):
        result=self.getNumpyOutputFromModel(inputForModel)
        fig = plt.figure(figsize=(6, 3.2))
        plt.imshow(result,label="Trénovacia sada")
        plt.show()

def getDatasetDicts():
    pclFileNames=os.listdir("./pclFiles")
    gtFileNames=os.listdir("./GroundTruth")
    pclDict={}
    gtDict={}
    listOfIDs=[]
    for pclFile in pclFileNames:
        fullName="./pclFiles/"+pclFile
        if(os.path.isfile(fullName)):
            key=fullName[11:-7]
            listOfIDs.append(key)
            pclDict.update({key:fullName})

    for gtFile in gtFileNames:
        key=gtFile[:-7]
        fullName="./GroundTruth/"+gtFile
        gtDict.update({key:fullName})
    return pclDict,listOfIDs,gtDict

#comparing input and output from network
def showImages(input, output):
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1,2, 1)
    plt.imshow(input,label="Trénovacia sada")
    plt.subplot(1, 2, 2)
    plt.imshow(output,label="Trénovacia sada")
    plt.show()


print("Results generated from model with ",numOfClasses, "classes")


def getFileDicts():
    listOfIDs=[]
    gtDict={}
    pclDict={}
    pclFileNames=os.listdir("./pclFiles/")
    gtFileNames=os.listdir("./GroundTruth")
    for pclFile in pclFileNames:
        fullName="./pclFiles/"+pclFile
        if(os.path.isfile(fullName)):
            key=fullName[11:-4]
            listOfIDs.append(key)
            pclDict.update({key:fullName})
    for gtFile in gtFileNames:
        for gtFile in gtFileNames:
            key=gtFile[:-7]
            fullName="./GroundTruth/"+gtFile
            gtDict.update({key:fullName})
    return pclDict,listOfIDs,gtDict

pclDict,listOfIDs,gtDict=getFileDicts()
network=modelWorker(modelName)

#if want to see the generated results and their ground truth images set showResults to True
showResults=False

accSum=0
maxFSum=0
samples=0
maxFMin=100
maxFMinKey=""
maxFMaxKey=""
maxFMax=0
cuda0=toolsForNetwork.get_device()

for key in listOfIDs:
    #only not rotated
    if(key[-2]!='0' and key[-2]!='1'):
        continue
    samples+=1
    print(samples,key)
    pclFileName=pclDict[key]
    gtName=gtDict[key]
    gt=torch.tensor(numpy.load(gtName))
    inputForNetwork=inputForModel(pclFileName)
    outputFromNetworkToShow=network.getNumpyOutputFromModel(inputForNetwork.tensorForModel)
    outputFromNetwork=network.tensorOutput
    acc,maxF=accuracyCalc.accuracy(outputFromNetwork,gt,cuda0)
    if(maxF>maxFMax):
        maxFMax=maxF
        maxFMaxKey=key
    if(maxF<maxFMin):
        maxFMin=maxF
        maxFMinKey=key
    accSum+=acc
    maxFSum+=maxF
    if(showResults):
        showImages(groundTruthImage,outputFromNetworkToShow)

print("test", (accSum/samples)*100, (maxFSum/samples)*100, " best predicted file: ",maxFMaxKey," worst predicted file: ",maxFMinKey)
