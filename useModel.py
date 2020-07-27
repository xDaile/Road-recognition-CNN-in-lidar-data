#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
import Model
import re
import parameters
import math
import ctypes

class inputForModel():
    def __init__(self,nameOfPCLfile):
        super(inputForModel,self).__init__()
        print("Started work with",nameOfPCLfile)
        self.loadFile(nameOfPCLfile)
        self.sortPointsInArea()
        self.pointsSortedInGrid= [[[] for i in range(200)] for j in range(400)]
        self.sortPointsInSelectedAreaIntoGrid()
        self.stats=[]
        self.countStats()
        self.tensorForModel=[]
        self.createTensor()
        #print(self.tensorForModel)


    def loadFile(self,nameOfPCLfile):
        try:
            self.rawFile = open(nameOfPCLfile, "r").readlines()
        except:
            print("file with input for model not found")
            exit(1)
        self.checkMetaInfo()
        self.rawPoints=self.rawFile[11:]

    def checkMetaInfo(self):
        #TODO some checks about file
        self.metaInfo=self.rawFile[0:11]
        metaArray=[]
        for meta in self.metaInfo:
            metaArray.append(re.split(r"\s+", meta))
        self.checkPCD(metaArray[0])
        self.checkVersion(metaArray[1])
        self.checkFields(metaArray[2])
        self.checkSize(metaArray[3])
        self.checkType(metaArray[4])
        self.checkCount(metaArray[5])
        self.checkWidth(metaArray[6])
        self.checkHeight(metaArray[7])
        self.checkViewPoint(metaArray[8])
        self.checkPoints(metaArray[9])
        self.checkData(metaArray[10])

    def checkData(self,dataMeta):
        if(dataMeta[0]!="DATA" or dataMeta[1]!="ascii"):
            print("Something wrong with eleventh - last line of the metainfo - bad format of data - use only ascii formats please")
            exit(1)

    def checkPoints(self,pointsMeta):
        if(pointsMeta[0]!="POINTS" or int(pointsMeta[1])<0):
            print("Something bad with tenth line of metainfo - bad format of POINTS")
            exit(1)

    def checkViewPoint(self,viewPointMeta):
        if(viewPointMeta[0]!="VIEWPOINT"):
            print("Something wrogh with nineth line of metainfo - bad format of viewPoint ")
            exit(1)

    def checkHeight(self,heightMeta):
        if(heightMeta[0]!="HEIGHT" or int(heightMeta[1])<0 ):
            print("Something wrong with eighth line of metainfo - bad format of HEIGHT of points - should be WIDTH and positive number")
            exit(1)

    def checkWidth(self,widthMeta):
        if(widthMeta[0]!="WIDTH" or int(widthMeta[1])<0):
            print("Something wrong with seventh line of metainfo - bad format of WIDTH of points - should be WIDTH and positive number")
            exit(1)

    def checkCount(self,countMeta):
        if(countMeta[0]!="COUNT" or countMeta[1]!="1" or countMeta[2]!="1" or countMeta[3]!="1" or countMeta[4]!="1" ):
            print("Something wrong with sixthline of metainfo - bad count of elemental data fields - use 1")
            exit(1)

    def checkType(self,typeMeta):
        if(typeMeta[0]!="TYPE" or typeMeta[1]!="F" or typeMeta[2]!="F" or typeMeta[3]!="F" or typeMeta[4]!="F" ):
            print("Something wrong with fifth line of metainfo - bad type of elemental data fields - use float")
            exit(1)

    def checkSize(self,sizeMeta):
        if(sizeMeta[0]!="SIZE" or sizeMeta[1]!="4" or sizeMeta[2]!="4" or sizeMeta[3]!="4" or sizeMeta[4]!="4" ):
            print("Something wrong with fourth line of metainfo - bad size of elemental data fields")
            exit(1)

    def checkFields(self,fieldsMeta):
        if(fieldsMeta[0]!="FIELDS" or fieldsMeta[1]!="x" or fieldsMeta[2]!="y" or fieldsMeta[3]!="z" or fieldsMeta[4]!="intensity"):
            print("Something wrong with third line of metainfo - bad fields description, use format which uses fields x,y,z,intensity")
            exit(1)

    def checkVersion(self,versionMeta):
        if(versionMeta[0]!="VERSION" or versionMeta[1]!="0.7"):
            print("Something wrong with second line in metainfo - probably bad version of pcl format - use 0.7 file format")
            exit(1)

    def checkPCD(self,pcdInfo):
        if(pcdInfo[0]!="#" or pcdInfo[1]!=".PCD"  or pcdInfo[2]!="v0.7" or pcdInfo[3]!="-" or pcdInfo[4]!="Point" or pcdInfo[5]!="Cloud" or pcdInfo[6]!="Data" or pcdInfo[7]!="file" or pcdInfo[8]!="format"):
            print("Something wrong with first line in metaInfo please use format PointXYZI")
            exit(1)

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
        #print(len(self.pointsInSelectedArea))
        #print(len(self.unusedPoints),len(self.pointsInSelectedArea))

    def sortPointsInSelectedAreaIntoGrid(self):
        i=0
        while i < len(self.pointsInSelectedArea):
            xCoord =int(399- int(((float(self.pointsInSelectedArea[i][0]))-6)*10))
            yCoord =int(199- int(((float(self.pointsInSelectedArea[i][1]))+10) * 10))
            self.pointsSortedInGrid[xCoord][yCoord].append(self.pointsInSelectedArea[i])
            i = i + 1

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

class modelWorker():
    def __init__(self,modelFileName):
        super(modelWorker,self).__init__()
        self.modelStructure=Model.Net()
        self.loadStateIntoStructure(modelFileName)

    def loadStateIntoStructure(self,modelFileName):
        self.savedModel=torch.load(modelFileName)
        self.modelStructure.load_state_dict(self.savedModel['model_state_dict'])
        self.modelStructure.eval()

    def useModel(self, inputForModel):
        outputFromModel=self.modelStructure(inputForModel)
        return outputFromModel

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
                    out[i][j]=1#,numpyAr[2][i][j])
                else:
                    if(numpyAr[1][i][j]>numpyAr[2][i][j]):
                        out[i][j]=0
                    else:
                        out[i][j]=2
                #else:
                #    out=max(numpAr[0][i][j],numpAr[1][i][j])
                j=j+1
            i=i+1
        return out
    def showResultFromNetwork(self,inputForModel):
        result=self.getNumpyOutputFromModel(inputForModel)
        fig = plt.figure(figsize=(6, 3.2))
        plt.imshow(result,label="Trénovacia sada")
        plt.show()
        return result

def getColor(classForCell):
    if(classForCell==0):

        return 0x1000FF00
    if(classForCell==1):
        return 0x10FF0000
    if(classForCell==2):
        return 0x1000FF00
    if(classForCell==3):
        return 255  <<16 | 255 <<8 | 255

def writeColoredPointCloud(pointsToColor,unusedPoints,nameOFWrittenFile):

    i=0
    newPoints=[]
    pointsCount=0
    while(i<400):
        j=0
        while(j<200):
            gridCell=pointsToColor[i][j]
            classForCell=outputFromNetwork[i][j]
            for point in gridCell:
                pointsCount+=1
                rgb=getColor(classForCell)
                newPoint=point[0] + " " + point[1]+ " " + point[2] + " " + str(rgb)  + "\n"
                newPoints.append(newPoint)
            j+=1
        i+=1

    for unusedPoint in unusedPoints:
        rgb=0x00000000
        newPoint=unusedPoint[0] + " " + unusedPoint[1]+ " " + unusedPoint[2] + " " + str(rgb) + "\n"
        newPoints.append(newPoint)
    pointsCount=pointsCount+len(unusedPoints)
    saveColoredPCD(newPoints,nameOFWrittenFile,pointsCount)

def saveColoredPCD(newPoints,nameOFWrittenFile,pointsCount):
    newFile="# .PCD v0.7 - Point Cloud Data file format\n" + \
            "VERSION 0.7\n"                                + \
            "FIELDS x y z rgb \n"                         + \
            "SIZE 4 4 4 4 \n"                           + \
            "TYPE F F F F \n"                           + \
            "COUNT 1 1 1 1 \n"                          + \
            "WIDTH " + str(pointsCount) + "\n"              + \
            "HEIGHT 1\n"                                   + \
            "VIEWPOINT 0 0 0 1 0 0 0\n"                    + \
            "POINTS " + str(pointsCount) + "\n"              + \
            "DATA ascii\n"
    pointsData=""
    for pointStr in newPoints:
        pointsData+=pointStr
    newFile=newFile+pointsData
    nameOfNewFile=nameOFWrittenFile+".pcd"
    f = open(nameOfNewFile, "w")
    f.write(newFile)
    f.close()
    print("file with colored point cloud  was written to ",nameOFWrittenFile,".pcd")


if(len(sys.argv)!=2 or sys.argv[1]=="-h"):
    print("usage: ./useModel file.pcl\n or ./useModel file.bin" )
    print("if you want to use another model, replace Model.tar in current directory")
    exit(1)
modelFileName="./Model.tar"
nameOfPCL=sys.argv[1]
if(nameOfPCL.find(".bin")>0 or nameOfPCL.find(".pcd")>0 ):
    if(nameOfPCL.find(".bin")>0):
        conversionProgramCmd="./kitti2pcl/bin/kitti2pcd --infile "+nameOfPCL+" --outfile ./tmp.pcd"
        os.system(conversionProgramCmd)
        inputClass=inputForModel("./tmp.pcd")
        os.system("rm ./tmp.pcd")
    else:
        inputClass=inputForModel(nameOfPCL)

    #inputClass=inputForModel("./pclFiles/um_000085.poinCL")
    network=modelWorker(modelFileName)
    outputFromNetwork=network.showResultFromNetwork(inputClass.tensorForModel)
    writeColoredPointCloud(inputClass.pointsSortedInGrid,inputClass.unusedPoints,"result")
    exit(0)
exit(1)
