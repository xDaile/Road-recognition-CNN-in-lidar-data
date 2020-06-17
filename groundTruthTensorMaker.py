#!/usr/bin/env python3

import os
import torch
import cv2
import parameters

'''
Transforming data from groundTruth matrices into one tensor
'''

def transformGT(loadedGroundTruth):
    i=0 #line index
    j=0 #item in line index
    road=[] #new matrix


    for line in loadedGroundTruth:#400 times will do
        addRoadRow=[]
        for matrix in line:#200 times will do
            if(matrix[0]==1):
                #value 1 is road
                    addRoadRow.append(1)
            else:
                if(matrix[0]==2):
                    #value 2 is not road
                    addRoadRow.append(0)
                else:
                    #value 3 - not computed
                    addRoadRow.append(3)
            j=j+1
        road.append(addRoadRow)
        addRoadRow=[]
        i=i+1
    return torch.tensor(road)


# Unused - will show the loaded image
# uncomment lines 49 or 59 for use
def showImg(img):
    y=y*10
    cv2.imshow("s",y)
    cv2.waitKey(5000)

def main():
    gtTest=os.listdir(parameters.groundTruthTestFilesFolder)
    gtTrain=os.listdir(parameter.groundTruthTrainFilesFolder)

    for file in gtTest:
        nameOfFileToOpen=parameters.groundTruthTestFilesFolder+file
        y = cv2.imread(nameOfFileToOpen)
        #showImg(y)
        y= transformGT(y)
        newName=parameters.groundTruthTestTensorsFolder+file[:-4]
        print(newName)
        torch.save(y,newName)

    for file in gtTrain:
        nameOfFileToOpen=parameter.groundTruthTrainFilesFolder+file
        y = cv2.imread(nameOfFileToOpen)
        y= transformGT(y)
        #showImg(y)
        newName=parameters.groundTruthTestTensorsFolder+file[:-4]
        print(newName)
        torch.save(y,newName)
