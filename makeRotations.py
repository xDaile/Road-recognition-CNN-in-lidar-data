#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import numpy
import getFiles
import parameters
import torch
import sys
#import time
import subprocess
import DatasetListForRotations
import multiprocessing

class pointCloud():
    def __init__(self,pclName,gtName):
        self.loadPclFile(pclName)
        self.loadGT(gtName)
        self.fileFormat=self.pointCloud[0]
        self.version=self.pointCloud[1]
        self.fields=self.pointCloud[2]
        self.size=self.pointCloud[3]
        self.type=self.pointCloud[4]
        self.count=self.pointCloud[5]
        self.width=self.pointCloud[6]
        self.heigth=self.pointCloud[7]
        self.viewPoint=self.pointCloud[8]
        self.points=self.pointCloud[9]
        self.data=self.pointCloud[10]
        self.pointsRaw=self.pointCloud[11:]
        self.pointsArray=[]
        self.pointsWithClass=[]
        self.newNameOfPCL=pclName.replace("pclFiles/","pclFiles/pclFilesWithClasses/")
        self.createdPoints=0

    # from raw format extract points to array
    def rawPointsToArray(self):
        for point in self.pointsRaw:
            self.fromRawPointToArray(point)

    #get classes from GT to the points
    def getClassForPoints(self):
        for point in self.pointsArray:
            self.getClassForPoint(point)

    #transform raw format into array format
    def fromRawPointToArray(self,point):

        newPoint=point[:-1]
        newPoint=newPoint.split(' ')
        self.pointsArray.append(newPoint)

    # get class from GT, for point, if it out of the grid(400x200) - not in the GT, third class will be asigned,
    def getClassForPoint(self,point):
        xGT=self.getXCoord(point[0])
        yGT=self.getYCoord(point[1])
        if(xGT<parameters.xDownBoundary or yGT<parameters.yDownBoundary or xGT>parameters.xUpBoundary or yGT>parameters.yUpBoundary):
            #if i want 4 classes(one for new areas, change 3 for some new number)
            classForPoint=3
        else:
            classForPoint=self.gt[xGT][yGT]
        newPoint=str(str(point[0])+ " " +str(point[1])+ " " +str(point[2])+ " " +str(point[3])+ " " +str(classForPoint))+" 0\n"
        self.pointsWithClass.append(newPoint)

    def getInverseXYCoords(self,i,j):
        x=-0.1*i+45.95
        y=-0.1*j+9.95
        #    y=10-(j/10)
#        if(y<0):
#            y=y-0.01

        return x,y

#TEST THAT and run that, also make more points into the corners
    def getGTintoPoints(self):
        i=0
        j=0
        while(i<400):
            while(j<200):
                classOfGT=self.gt[i][j]
                #find coef for left, right corner, midle, up, down etc
                x,y=self.getInverseXYCoords(i,j)

                newPoints=[str(str(x+0.25)+" "+ str(y+0.25)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x+0.25)+" "+ str(y-0.25)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x-0.25)+" "+ str(y+0.25)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x-0.25)+" "+ str(y-0.25)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x)+" "+ str(y)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n"]
                         # str(str(x+0.75)+" "+ str(y)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n"]
                self.createdPoints=self.createdPoints+5
                for newPoint in newPoints:
                    self.pointsWithClass.append(newPoint)

                j=j+1
                #break
            i=i+1
            j=0


    #loading ground truth, transform it into array ready for work,
    #assing the value that is saved in GT, not [x,x,x] because, that will be done after rotation when gt will be created
    def loadGT(self,gtName):
        gtRaw=numpy.load(gtName)
        gtLine=[]
        self.gt=[]
        for line in gtRaw:
            for matrix in line:
                gtLine.append(matrix[0])
            self.gt.append(gtLine)
            gtLine=[]

    # loading pcl file
    def loadPclFile(self,pclName):
        self.pointCloud=open(pclName, "r").readlines()

    #transform x-coord of point into second position in 2d array 400(this position) x 200
    def getXCoord(self,xOrig):
        if (float(xOrig) > parameters.xDownBoundary and float(xOrig) < parameters.xUpBoundary):
            return int(399- int(((float(xOrig))-6)*10))
        else:
            return -1

    #transform y-coord of point into second position in 2d array 400 x 200(this position)
    def getYCoord(self,yOrig):
        if (float(yOrig) > parameters.yDownBoundary and float(yOrig) < parameters.yUpBoundary):
            return int(199- int(((float(yOrig))+10) * 10))
        return 0

    def saveWithCLass(self):
        #self.newName=self.pclName.replace("pclFiles/","pclFiles/pclFilesWithClasses/")
        with open(self.newNameOfPCL, 'w') as f:
            #self.newNameOfPCL=newName
            f.write(self.fileFormat)
            f.write(self.version)
            newFields=self.fields[:-1]+" intensity_variance height_variance\n"
            f.write(newFields)

            newSize=self.size[:-1]+" 4 4\n"

            f.write(newSize)
            newType=self.type[:-1]+" F F\n"

            f.write(newType)
            newCount=self.count[:-1]+" 1 1\n"

            f.write(newCount)

            #       old width sing + str(old number of width + added points )
            newWidth=self.width[0:6]+str(int(self.width[-7:])+self.createdPoints)+"\n"
            f.write(newWidth)


            f.write(self.heigth)
            f.write(self.viewPoint)
            #              old points sing + str(old number of points + added points )
            newPointsCount=self.points[0:7]+str(int(self.points[-8:])+self.createdPoints)+"\n"
            f.write(newPointsCount)

            f.write(self.data)
            for point in self.pointsWithClass:
                f.write(point)

    def pointCloudForRotation(self):
        scalarPoints=[]
        for point in self.pointsArray:
            scalarPoints.append((point[0],point[1]))
        return scalarPoints#,yScalar






def createRotation(pclName):
    cmd="./rotator/rotator " + pclName
    os.system(cmd)

def buildRotator():
    buildCmd="(cd ./rotator/ && make)"
    os.system(buildCmd)
    changeRights="(cd ./rotator/ && chmod +777 rotator)"
    os.system(changeRights)

def multiprocessFunction(pclNameANDgtName):
    pclName,gtName=pclNameANDgtName;
    print("Created Process for:",pclName)
    pclCloud=pointCloud(pclName,gtName)
    #convert point cloud to array of points
    pclCloud.rawPointsToArray()

    #get class from GT into points where it is possible
    pclCloud.getClassForPoints()
    pclCloud.getGTintoPoints()
    pclCloud.saveWithCLass()
    #call c++ program for rotations of the point cloud
    createRotation(pclCloud.newNameOfPCL)


dataset=DatasetListForRotations.DatasetList()

#build c++ program for rotations of clouds
buildRotator()

#shorten dataset EDIT
#dataset.itemsList=dataset.itemsList[0:6]

#left 2 proccesors for other things
usableProcessors=multiprocessing.cpu_count()-2

pool = multiprocessing.Pool(processes=usableProcessors)
pool.map(multiprocessFunction, dataset.itemsList)
