#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   Author: Michal Zelenak
   BUT Faculty of Information Technology
   This is code written for the bachelor thesis
   Project: Object Detection in the Laser Scans Using Convolutional Neural Networks
"""

import os
import sys
import re
import math
import pandas as pd
import parameters
import torch
import numpy as np
import multiprocessing
from notify_run import Notify
import pointsWorker as pLIB

notify = Notify()

def saveToCsv(variable, name):
    Csv = pd.DataFrame(variable)
    Csv.to_csv(name, index=None, header=False)

#this function flip ground truth tensor by x and save it
def flipGTByXandSave(tensor,nameOfGT):
    #into array
    tensorInArray=tensor.numpy()
    i=0
    #create empty array
    flipped= [[0] * 200] * 400
    while(i<400):
        #flip lines one by one
        flipped[i]=np.flip(tensorInArray[i])
        i=i+1
    tensorToSave=torch.tensor(flipped)
    #get new name - flip first number
    shift=0
    if(nameOfGT.find("umm")>0):
        shift=1
        #Example of what this line is doing
                                                        #            to here is 29 ->|  |110076 - flag 1 sign that data were flipped over vertical axis
    newName=nameOfGT[0:29+shift]+"1"+nameOfGT[30+shift:] #./Dataset/gtTensors/train/umm_010076
    torch.save(tensorToSave,newName)
#Example of what this line is doing
#this will save ground truth into Dataset folder and also create and save its flipped by x version
def saveGroundTruth(groundTruth,nameOfPCL):
    #create new name, also change location of created file
    if(nameOfPCL[-6]=="0" or nameOfPCL[-6]=='1'):
        newName=parameters.gtTestTensors+nameOfPCL[0:-4]
    else:
        newName=parameters.gtTrainTensors+nameOfPCL[0:-4]
    tensor=torch.tensor(groundTruth)
    torch.save(tensor,newName)
    flipGTByXandSave(tensor,newName)

#flip the tensor by x axis and save the tensor into dataset folder
def flipDataByXAndSave(tensor,nameOfSaved):
    shift=0
    if(nameOfSaved.find("umm")>0):
        shift=1

    #Example of what this line is doing
    #convert ./Dataset/gtTensors/train/uu_023064 to ./Dataset/gtTensors/train/uu_123064
    newName=nameOfSaved[0:len(parameters.gtTrainTensors)+shift]+"1"+nameOfSaved[len(parameters.gtTrainTensors)+1+shift:]
    tensorInArray=tensor.numpy()
    statIndex=0
    newFlippedTensor=[]
    while(statIndex<6):
        workWith=tensorInArray[statIndex]
        flippedStat=[]
        for line in workWith:
            flippedLine=np.flip(line)
            flippedStat.append(flippedLine)
        newFlippedTensor.append(torch.tensor(flippedStat))
        statIndex=statIndex+1
    newFlipped=torch.stack(newFlippedTensor)
    torch.save(newFlipped,newName)

#this will save the created tensor, and also his by x flipped version
def saveTensor(tensor,nameOfPCL):
    shift=0
    if(nameOfPCL.find("umm")>0):
        shift=1
    if(nameOfPCL[-6]=="0" or nameOfPCL[-6]=="1"):
        newName=parameters.testTensorFolder+nameOfPCL[0:-4]
    else:
        newName=parameters.trainTensorFolder+nameOfPCL[0:-4]
    torch.save(tensor,newName)
    flipDataByXAndSave(tensor,newName)

def getEmpty2DArray(firstDim,secondDim):
    return [[[] for i in range(secondDim)] for j in range(firstDim)]


def saveStats(density,maxEL,minEL,avgRefList,avgELList,stdEL):
    saveStats(density,maxEL,minEL,avgRefList,avgELList,stdEL)
    saveToCsv(density, newName + "_density.csv")
    saveToCsv(maxEL, newName + "_maxEl.csv")
    saveToCsv(minEL, newName + "_minEL.csv")
    saveToCsv(avgRefList, newName + "_meanRef.csv")
    saveToCsv(avgELList, newName + "_meanEL.csv")
    saveToCsv(stdEL, newName + "_stdEL.csv")

class pclFileClass():
    def __init__(self,nameOfPCL):
        super(pclFileClass,self).__init__()
        print("Generating of input and GroundTruth from",nameOfPCL)
        self.getFilePathToPCL(nameOfPCL)
        self.loadPCL()
        self.pointsInGrid=getEmpty2DArray(400,200)
        self.sortPointsToGrid()

    def loadPCL(self):
        pclFile = open(self.filePathToPCL, "r").readlines()
        self.pclHeader=pclFile[0:11]
        arrayOfAsciiPoints = pclFile[11:]
        self.unsortedPoints=[]
        for oneAsciiPoint in arrayOfAsciiPoints:
            # split items by every whitespace
            onePointArray = re.split(r"\s+",oneAsciiPoint)

            # last item in list is only whitespace for newline
            onePointArray = onePointArray[:6]
            # check if the point is in the area 400x200 metres that we will consider for counting stats
            if (pLIB.checkPointsBoundary(onePointArray)):
                self.unsortedPoints.append(onePointArray)

    def maximumOfClasses(self,points):
        gtArray=[]
        for point in points:
            gtArray.append(pLIB.getCPointCoordinate(point))
        #road
        class0=0
        #not road
        class1=0
        #lidar not saw this
        class2=0
        #not projected
        class3=0
        #count classes
        for item in gtArray:
            if(item=='0'):
                class0=class0+1
            if(item=='1'):
                class1=class1+1
            if(item=='2'):
                class2=class2+1
            if(item=='3'):
                class3=class3+1
        #check if class0 or class2 have some points
        if(class0+class1>0):
            if(class0>class1):
                return 0
            if(class1>=class0):
                return 1
        #here is the effort to make as few class 3 points as possible
        if(class2>class3):
            return 2
        return 3

    def createTensor(self,arrayValues):
        statsAboutFile=[]
        for oneStat in arrayValues:
            torch_tensor=torch.tensor(oneStat)
            torch_tensor=torch_tensor.float()
            statsAboutFile.append(torch_tensor)
        return torch.stack(statsAboutFile)

    def getFilePathToPCL(self,nameOfPCL):
        self.filePathToPCL= parameters.rotatedPCLFiles+nameOfPCL


    # transforming the list of points into grid structure(each cell is 0.1*0.1 cm) -> array 400x200
    # 6 to 46 -> 40metres, -10 to 10 ->20 metres, grid is 0.1*0.1 metres, so it will be 400*200 array
    #                       400metres              0.1m
    # --|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|0.1m
    # --|--|--|................................|--|--|
    # --|--|--|................................|--|--|   200 metres
    # --|--|--|................................|--|--|
    # --|--|--|................................|--|--|
    def sortPointsToGrid(self):
        for point in self.unsortedPoints:
            xCoord =int(399- int(((float(pLIB.getXPointCoordinate(point)))-6)*10))
            yCoord =int(199- int(((float(pLIB.getYPointCoordinate(point)))+10) * 10))
            self.pointsInGrid[xCoord][yCoord].append(point)

    def sortOriginalAndAddedPoints(self,cellInGrid):
        self.originalPoints=[]
        self.addedPoints=[]
        for point in cellInGrid:
            #check the Flag for false point
            if(point[5]!='1'):
                self.originalPoints.append(point)
            else:
                self.addedPoints.append(point)

    def computeStandartDeviation(self,originalPoints,avgELSum):
        std=0
        for point in originalPoints:
            std = std + ((float(pLIB.getZPointCoordinate(point)) - avgELSum) * (float(pLIB.getZPointCoordinate(point)) - avgELSum))
        std = std / len(originalPoints)
        std = math.sqrt(std)
        return std

    def getStats(self):
        density     = getEmpty2DArray(400,200)
        minEL       = getEmpty2DArray(400,200)
        maxEL       = getEmpty2DArray(400,200)
        avgELList   = getEmpty2DArray(400,200)
        avgRefList  = getEmpty2DArray(400,200)
        stdEL       = getEmpty2DArray(400,200)
        self.groundTruth = getEmpty2DArray(400,200)
        i = 0
        # going throught each cell, and counting the stats in each cell
        while i < 400:
            j=0
            while j < 200:
                cellInGrid = self.pointsInGrid[i][j]

                # if the cell is empty
                if (len(cellInGrid) == 0.0):
                    minEL[i][j] = 0.0
                    maxEL[i][j] = 0.0
                    avgELList[i][j] = 0.0
                    avgRefList[i][j] = 0.0
                    density[i][j] = 0.0
                    stdEL[i][j] = 0.0
                    self.groundTruth[i][j]=parameters.ClassForPointOutOfRotation
                    #should propagate from surrounding, or let it be 3

                # points are [x,y,z,r], where x is x coordinate, y is y coordinate, z is elevation and r is reflectivity
                #there are some points in the cell of grid
                else:
                    self.sortOriginalAndAddedPoints(cellInGrid)

                    #if there are only added points
                    if(len(self.originalPoints)==0):
                        minEL[i][j]=0
                        maxEL[i][j]=0
                        avgELList[i][j]=0
                        avgRefList[i][j] = 0.0
                        density[i][j] = 0.0
                        stdEL[i][j] = 0.0
                        self.groundTruth[i][j]=self.maximumOfClasses(self.addedPoints)

                    #there are some original points
                    else:
                        # setting the minimum and maximum with the first point in the cell(because we want to have something to compare to)
                        minEL[i][j]=float(pLIB.getZPointCoordinate(self.originalPoints[0]))
                        maxEL[i][j]=float(pLIB.getZPointCoordinate(self.originalPoints[0]))
                        numOfPointsInCell=len(self.originalPoints)
                        density[i][j]=numOfPointsInCell
                        # average elevation
                        avgELSum=0.0
                        # average reflectivity
                        avgRefSum=0.0


                        # going throught the list of original points in the cell

                        for point in self.originalPoints:
                            # actual elevation
                            elevation = float(pLIB.getZPointCoordinate(point))

                            # summary of elevations(z coordinate)
                            avgELSum = avgELSum + elevation

                            # summary of reflectivity(r coordinate)
                            avgRefSum = avgRefSum + float(pLIB.getIPointCoordinate(point))

                            # finding lowest and highest points
                            minEL[i][j]=min(minEL[i][j], elevation)
                            maxEL[i][j]=max(maxEL[i][j], elevation)

                        # counting the mean of elevation and reflectivity
                        avgELList[i][j] = avgELSum / numOfPointsInCell
                        avgRefList[i][j] = avgRefSum / numOfPointsInCell

                        # standard deviation of elevation
                        # for counting the standard deviation we have to count mean elevation first first and then standard deviation)
                        stdEL[i][j]=self.computeStandartDeviation(self.originalPoints,avgELSum)

                        #added points are better distributed -> count gt from those points
                        if(len(self.addedPoints)==0 and len(self.originalPoints)!=0):
                            self.groundTruth[i][j]=self.maximumOfClasses(self.originalPoints)
                        if(len(self.addedPoints)!=0):
                            self.groundTruth[i][j]=self.maximumOfClasses(self.addedPoints)
                        else:
                            self.groundTruth[i][j]=parameters.ClassForPointOutOfRotation
                j += 1
            i += 1
        return self.createTensor([density,maxEL,avgELList,avgRefList,minEL,stdEL])

    def getGroundTruth(self):
        return self.groundTruth




def createTensorAndGTFromFile(nameOfPCL):
    pcl=pclFileClass(nameOfPCL)
    tensorForSave=pcl.getStats()
    groundTruth=pcl.getGroundTruth()
    saveGroundTruth(groundTruth,nameOfPCL)
    saveTensor(tensorForSave,nameOfPCL)
    if(len(sys.argv)>1):
        if(sys.argv[1]=="-saveStats"):
            newName=parameters.rotatedStats+nameOfPCL[22:-7]

def main():
    if(len(sys.argv)>1):
        if(sys.argv[1]=="-help" or sys.argv[1]=="--help"):
            print("Use ./rotatedPclToTensors.py\nif you want stats to be generated use ./rotatatedPclToTensors.py -saveStats")
            exit(0)
    rotatedFiles=os.listdir(parameters.rotatedPCLFiles)
    usableProcessors=multiprocessing.cpu_count()-2

    #comment next line if not debugging
    #createTensorAndGTFromFile("umm_010076.poinCL")
    #exit(1)
    pool = multiprocessing.Pool(processes=usableProcessors)
    pool.map(createTensorAndGTFromFile, rotatedFiles)

if __name__ == "__main__":
    main()
    notify.send("Dataset created")
