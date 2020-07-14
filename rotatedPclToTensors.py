#!/usr/bin/env python3
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

notify = Notify()
# booundaries(area) where points will be counted
xDownBoundary = parameters.xDownBoundary
xUpBoundary = parameters.xUpBoundary
yDownBoundary = parameters.yDownBoundary
yUpBoundary = parameters.yUpBoundary

def saveToCsv(variable, name):
    Csv = pd.DataFrame(variable)
    Csv.to_csv(name, index=None, header=False)

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
    newName=nameOfGT[0:29+shift]+"1"+nameOfGT[30+shift:]
    torch.save(tensorToSave,newName)


def saveGroundTruth(groundTruth,nameOfPCL):
    #create new name, also change location of created file
    if(nameOfPCL[-9]=="0" or nameOfPCL[-9]=='1'):
        newName=parameters.gtTestTensors+nameOfPCL[22:-7]
    else:
        newName=parameters.gtTrainTensors+nameOfPCL[22:-7]
    gt=[]
    gtLine=[]
    for line in groundTruth:
        gtLine=[]
        for item in line:
            if(item==0):
                gtLine.append(0)
            else:
                if(item==1):
                    gtLine.append(1)
                else:
                    gtLine.append(2)
        gt.append(torch.tensor(gtLine))
    tensor=torch.stack(gt)
    torch.save(tensor,newName)
    flipGTByXandSave(tensor,newName)

def flipDataByXAndSave(tensor,nameOfSaved):
    shift=0
    if(nameOfSaved.find("umm")>0):
        shift=1

    newName=nameOfSaved[0:26+shift]+"1"+nameOfSaved[27+shift:]
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

def saveTensor(tensor,nameOfPCL):

    shift=0
    if(nameOfPCL.find("umm")>0):
        shift=1
    if(nameOfPCL[-9]=="0" or nameOfPCL[-9]=="1"):
        newName=parameters.testTensorFolder+nameOfPCL[22:-7]
    else:
        newName=parameters.trainTensorFolder+nameOfPCL[22:-7]
    torch.save(tensor,newName)
    flipDataByXAndSave(tensor,newName)

def createTensor(arrayValues):
    statsAboutFile=[]
    for oneStat in arrayValues:
        torch_tensor=torch.tensor(oneStat)
        torch_tensor=torch_tensor.float()
        statsAboutFile.append(torch_tensor)
    return torch.stack(statsAboutFile)

def maximumOfClasses(points):
    gtArray=[]
    for point in points:
        gtArray.append(point[4])
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
    #return 2 #comment THIS LINE IF WANT TO HAVE 0123 in tensors, otherwise, there will be just 012
    if(class2>class3):
        return 2
    return 3

def createTensorAndGTFromFile(nameOfPCL):
    nameOfPCL=parameters.rotatedPCLFiles+nameOfPCL
    print("Started work with",nameOfPCL)
    # opening and reading the file
    m = open(nameOfPCL, "r").readlines()

    # first eleven lines are metainfo
    m = m[11:]

    # index variable
    i = 0

    # here we will store the points
    new_m = []

    # selecting the points which we will count stats from
    while i < len(m):

        # split items by every whitespace
        s = re.split(r"\s+", m[i])

        # last item in list is only whitespace for newline
        s = s[:6]
        # check if the point is in the area 400x200 metres that we will consider for counting stats
        if (
            float(s[0]) > xDownBoundary
            and float(s[0]) < xUpBoundary
            and float(s[1]) > yDownBoundary
            and float(s[1]) < yUpBoundary
        ):
            new_m.append(s)
        i = i + 1

    # creating 400x200 empty array
    x = [[[] for i in range(200)] for j in range(400)]

    i = 0

    # transforming the list of points into grid structure(each cell is 0.1*0.1 cm) -> array 400x200
    # 6 to 46 -> 40metres, -10 to 10 ->20 metres, grid is 0.1*0.1 metres, so it will be 400*200 array
    #
    #    ->>> |--|  <<<- to field like this points are sorted

    #                       400metres                            0.1m
    # --|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|0.1m
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|   200 metres
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|.........................................|--|--|--|
    # --|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|

    # creating the grid(2d array) by the x,y coordinates

    while i < len(new_m):
        xCoord =int(399- int(((float(new_m[i][0]))-6)*10))
        yCoord =int(199- int(((float(new_m[i][1]))+10) * 10))
        x[xCoord][yCoord].append(new_m[i])
        i = i + 1
    # creating empty 2d 400x200 arrays 6times, for each statistic one array
    density = [[0 for i in range(200)] for j in range(400)]
    minEL = [[0 for i in range(200)] for j in range(400)]
    maxEL = [[0 for i in range(200)] for j in range(400)]
    avgELList = [[0 for i in range(200)] for j in range(400)]
    avgRefList = [[0 for i in range(200)] for j in range(400)]
    stdEL = [[0 for i in range(200)] for j in range(400)]
    groundTruth = [[0 for i in range(200)] for j in range(400)]

    i = 0
    j = 0

    # going throught each cell, and counting the stats in each cell
    while i < 400:
        while j < 200:
            item = x[i][j]

            # counting the cell density
            #density[i][j] += float(len(item))
            # if the cell is empty
            if len(item) == 0.0:
                minEL[i][j] = 0.0
                maxEL[i][j] = 0.0
                avgELList[i][j] = 0.0
                avgRefList[i][j] = 0.0
                density[i][j] = 0.0
                stdEL[i][j] = 0.0
                groundTruth[i][j]=2 #EDIT
                #should propagate from surrounding, or let it be 3

            # points are [x,y,z,r], where x is x coordinate, y is y coordinate, z is elevation and r is reflectivity
            #there are some points
            else:
                #count original and added points
                originalPoints=[]
                addedPoints=[]
                for t in item:
                    if(t[5]!='1'):
                        originalPoints.append(t)
                    else:
                        addedPoints.append(t)
                #if there are only added points
                if(len(originalPoints)==0):
                    minEL[i][j]=0
                    maxEL[i][j]=0
                    avgELList[i][j]=0
                    avgRefList[i][j] = 0.0
                    density[i][j] = 0.0
                    stdEL[i][j] = 0.0
                    groundTruth[i][j]=maximumOfClasses(addedPoints)

                #there are some original points, 'else' would be enough but this is more beautifull
                if(len(originalPoints)!=0):
                    # setting the minimum and maximum with the first point in the cell(because we want to have something to compare to)
                    minEL[i][j]=float(originalPoints[0][2])
                    maxEL[i][j]=float(originalPoints[0][2])
                    numOfPointsInCell=len(originalPoints)
                    density[i][j]=numOfPointsInCell
                    # average elevation
                    avgELSum=0.0
                    # average reflectivity
                    avgRefSum=0.0
                    # standard deviation of elevation(variable for computing)
                    std = 0.0
                    # going throught the list of original points in the cell

                    for t in originalPoints:
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
                    for t in originalPoints:
                        std = std + ((float(t[2]) - avgELSum) * (float(t[2]) - avgELSum))
                    std = std / numOfPointsInCell
                    std = math.sqrt(std)

                    # saving the data
                    stdEL[i][j] = std
                    avgELList[i][j] = avgEL
                    avgRefList[i][j] = avgRef
                    #added points are better distributed -> count gt from those points
                    if(len(addedPoints)==0):
                        groundTruth[i][j]=maximumOfClasses(originalPoints)
                    else:
                        groundTruth[i][j]=maximumOfClasses(addedPoints)

            j = j + 1
        j = 0
        i = i + 1

    #points that work were done was bigger, but the values are only stats from points that were original and also for computation were used only x,y,z,r values
    #stats should be in this order saved, if we want to have same file as from ./gridmaker
    tensorForSave=createTensor([density,maxEL,avgELList,avgRefList,minEL,stdEL])
    saveGroundTruth(groundTruth,nameOfPCL)
    saveTensor(tensorForSave,nameOfPCL)

    if(len(sys.argv)>1):
        if(sys.argv[1]=="-saveStats"):
            saveToCsv(density, parameters.rotatedStats+nameOfPCL[22:-7] + "_density.csv")
            saveToCsv(maxEL, parameters.rotatedStats+nameOfPCL[22:-7] + "_maxEl.csv")
            saveToCsv(minEL, parameters.rotatedStats+nameOfPCL[22:-7] + "_minEL.csv")
            saveToCsv(avgRefList, parameters.rotatedStats+nameOfPCL[22:-7] + "_meanRef.csv")
            saveToCsv(avgELList, parameters.rotatedStats+nameOfPCL[22:-7] + "_meanEL.csv")
            saveToCsv(stdEL, parameters.rotatedStats+nameOfPCL[22:-7] + "_stdEL.csv")

def main():
    if(len(sys.argv)>1):
        if(sys.argv[1]=="-help" or sys.argv[1]=="--help"):
            print("Use ./rotatedPclToTensors.py\nif you want stats to be generated use ./rotatatedPclToTensors.py -saveStats")
            exit(0)
    rotatedFiles=os.listdir(parameters.rotatedPCLFiles)
    usableProcessors=multiprocessing.cpu_count()-2

    #comment next line if not debugging
#    createTensorAndGTFromFile("umm_010000.poinCL")
#    exit(1)
    pool = multiprocessing.Pool(processes=usableProcessors)
    pool.map(createTensorAndGTFromFile, rotatedFiles)

if __name__ == "__main__":
    main()
    notify.send("rotatedPclToTensors done")
