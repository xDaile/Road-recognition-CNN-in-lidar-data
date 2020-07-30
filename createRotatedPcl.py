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

import matplotlib.pyplot as plt
import os
import numpy
import getFileLists
import parameters
import torch
import sys
#import time
import subprocess
import DatasetListForRotations
import multiprocessing
from notify_run import Notify

notify = Notify()

#works as a loader of pclCloud, also while loading pcl file, it project ground truth data into point cloud
class pointCloudProjector():
    def __init__(self,pclName,gtName):
        super(pointCloudProjector,self).__init__()
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

    #transform raw format into array format
    def fromRawPointToArray(self,point):
        newPoint=point[:-1]
        newPoint=newPoint.split(' ')
        self.pointsArray.append(newPoint)

    #get classes from GT to the points
    def getClassForPoints(self):
        for point in self.pointsArray:
            self.getClassForPoint(point)

    # get class from GT, for point, if it out of the grid(400x200) - not in the GT, third class will be asigned,
    def getClassForPoint(self,point):
        xGT=self.getXCoord(point[0])
        yGT=self.getYCoord(point[1])
        if(xGT==-100 or yGT==-100):
            classForPoint=parameters.ClassForPointOutOfRotation #those are points in original point cloud for which no class is is_available
        else:
            classForPoint=self.gt[xGT][yGT]#012
        newPoint=str(str(point[0])+ " " +str(point[1])+ " " +str(point[2])+ " " +str(point[3])+ " " +str(classForPoint))+" 0\n"
        self.pointsWithClass.append(newPoint)

    #returns place in the point cloud depending on posistion in the array
    def getInverseXYCoords(self,i,j):
        x=-0.1*i+45.95
        y=-0.1*j+9.95
        return x,y

    #projection of ground truth data into points
    def projectGTintoPointCloud(self):
        i=0
        j=0
        while(i<400):
            while(j<200):
                classOfGT=self.gt[i][j]
                #find coef for left, right corner, midle, up, down etc
                x,y=self.getInverseXYCoords(i,j)
                newPoints=[str(str(x+0.025)+" "+ str(y+0.025)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x+0.025)+" "+ str(y-0.025)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x-0.025)+" "+ str(y+0.025)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x-0.025)+" "+ str(y-0.025)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n",
                          str(str(x)+" "+ str(y)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n"]
                         # str(str(x+0.75)+" "+ str(y)+" "+str(0)+" "+str(0)+" "+str(self.gt[i][j])+" "+str(1))+" 0\n"]
                self.createdPoints=self.createdPoints+5
                for newPoint in newPoints:
                    self.pointsWithClass.append(newPoint)
                j=j+1
            i=i+1
            j=0


    #loading ground truth image
    def loadGT(self,gtName):
        self.gt=numpy.load(gtName,allow_pickle=True)


    # loading pcl file
    def loadPclFile(self,pclName):
        self.pointCloud=open(pclName, "r").readlines()

    #transform x-coord of point cloud into first position in 2d array 400(this position) x 200
    def getXCoord(self,xOrig):
        if (float(xOrig) >= parameters.xDownBoundary and float(xOrig) <= parameters.xUpBoundary):
            return int(399- int(((float(xOrig))-6)*10))
        else:
            return -100

    #transform y-coord of point cloud into second position in 2d array 400 x 200(this position)
    def getYCoord(self,yOrig):
        if (float(yOrig) >= parameters.yDownBoundary and float(yOrig) <= parameters.yUpBoundary):
            return int(199- int(((float(yOrig))+10) * 10))
        return -100

    #save the new point cloud
    def saveWithCLass(self):
        #self.newName=self.pclName.replace("pclFiles/","pclFiles/pclFilesWithClasses/")
        with open(self.newNameOfPCL, 'w') as f:
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

            newPointsCount=self.points[0:7]+str(int(self.points[-7:])+self.createdPoints)+"\n"
            f.write(newPointsCount)

            f.write(self.data)
            for point in self.pointsWithClass:
                f.write(point)


def createRotatedVersionsOfPointCloud(pclName):
    cmd="./pclRotator/pclRotator " + pclName
    os.system(cmd)

#builds c++ program for transformation of the point cloud, this program can be found in directory ./pclRotator
def buildCppRotator():
    buildCmd="(cd ./pclRotator/ && make)"
    os.system(buildCmd)
    changeRights="(cd ./pclRotator/ && chmod +777 pclRotator)"
    os.system(changeRights)

#function for one processor
def multiprocessFunction(pclNameANDgtName):
    #pclNameANDgtName is tuple send to this function
    pclName,gtName=pclNameANDgtName;
    print("Created Process for:",pclName)
    #load ground truth images and pclFiles
    pclCloud=pointCloudProjector(pclName,gtName)

    #convert point cloud to array of points
    pclCloud.rawPointsToArray()

    #get classes from GT image into points where it is possible
    pclCloud.getClassForPoints()

    #prroject points into the cloud, also will create fake points
    pclCloud.projectGTintoPointCloud()

    #save created point cloud
    pclCloud.saveWithCLass()

    #call c++ program for rotations of the point cloud
    createRotatedVersionsOfPointCloud(pclCloud.newNameOfPCL)

def main():
    #get the dataset
    dataset=DatasetListForRotations.DatasetList()

    #build c++ program for rotations of clouds
    buildCppRotator()

    #let 2 proccesors be free for other things
    usableProcessors=multiprocessing.cpu_count()-2

    #uncoment next line for try once this program
    #multiprocessFunction(('./pclFiles/umm_000076.poinCL', './GroundTruth/umm_000076_gt.npy'))
    #exit(1)
    #multiprocessFunction(('./pclFiles/umm_000076.pcd', './GroundTruth/umm_000076_gt.npy'))
    #exit(1)

    #this two lines will ensure the multiprocessing
    pool = multiprocessing.Pool(processes=usableProcessors)
    pool.map(multiprocessFunction, dataset.itemsList)

if __name__ == "__main__":
    main()
    notify.send("Rotation of pcl Files done")
