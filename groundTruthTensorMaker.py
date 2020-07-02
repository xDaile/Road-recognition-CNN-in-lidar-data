#!/usr/bin/env python3

import os
import torch
import cv2
import parameters
import numpy as np

'''
Transforming data from groundTruth matrices into one tensor
'''

'''def transformGT(loadedGroundTruth):
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
    return torch.tensor(road)'''


def transform2GT(loadedGroundTruth):
    i=0 #line index
    j=0 #item in line index
    road=[]
    #class0=[] #new matrix
    #class1=[] #new matrix
    #class2=[] #new matrix


    for line in loadedGroundTruth:#400 times will do
        roadLine=[]
        #class1Line=[]
        #class2Line=[]

        for matrix in line:#200 times will do
            if(matrix[0]==1):
                #value 1 is road
                    roadLine.append([1,0,0])
                    #class1Line.append(0)
                    #class2Line.append(0)
            else:
                if(matrix[0]==2):
                    #value 2 is not road
                    roadLine.append([0,1,0])
                    #class1Line.append(1)
                    #class2Line.append(0)
                else:
                    #value 3 - not computed
                    roadLine.append([0,0,1])
                    #class1Line.append(0)
                    #class2Line.append(1)
        road.append(roadLine)
        #class1.append(class1Line)
        #class2.append(class2Line)
        roadLine=[]
        #class1Line=[]
        #class2Line=[]
    return torch.stack(road)

def flipByXOneDimension(tensor):
    tensorInArray=tensor.numpy()
    i=0
    flipped= [[0] * 200] * 400
    while(i<400):
        flipped[i]=np.flip(tensorInArray[i])
        i=i+1
    return torch.tensor(flipped)

def addNumberToNumberAtName(name,number):
    possibleNumber=[100,200,300]
    if(number in possibleNumber):
        endOfNameOfFile=name[-3:]
        nameWithoutEnd=name[:-3]
        fileWithoutNumber=nameWithoutEnd[:-3]
        fileNumber=int(nameWithoutEnd[-3:])
        if(fileNumber>99):
            return 0
        newFileNumber=fileNumber+number
        newName=fileWithoutNumber+str(newFileNumber)+endOfNameOfFile
        return (newName)
    else:
        print("that number cannot be assigned to name of the file")
        exit(1)

# Unused - will show the loaded image
# uncomment lines 49 or 59 for use
def showImg(img):
    img=img*30
    cv2.imshow("s",img)
    cv2.waitKey(50000)

def main():
    gtTest=os.listdir(parameters.groundTruthTestFilesFolder)
    gtTrain=os.listdir(parameters.groundTruthTrainFilesFolder)

    for file in gtTest:
        nameOfFileToOpen=parameters.groundTruthTestFilesFolder+file
        y = cv2.imread(nameOfFileToOpen)
        #showImg(y)
        y= transform2GT(y)
        newName=parameters.groundTruthTestTensorsFolder+file[:-4]
        print(newName)
        #exit(1)
        torch.save(y,newName)
        orig=torch.load(newName)
        flippedByX=flipByXOneDimension(orig)
        nameForXFlipped=addNumberToNumberAtName(newName,100)
        print(nameForXFlipped)
        torch.save(flippedByX,nameForXFlipped)


    for file in gtTrain:
        nameOfFileToOpen=parameters.groundTruthTrainFilesFolder+file
        y = cv2.imread(nameOfFileToOpen)
        y= transform2GT(y)
        #showImg(y)
        newName=parameters.groundTruthTrainTensorsFolder+file[:-4]
        print(newName)
        torch.save(y,newName)
        orig=torch.load(newName)
        flippedByX=flipByXOneDimension(orig)
        nameForXFlipped=addNumberToNumberAtName(newName,100)
        print(nameForXFlipped)
        torch.save(flippedByX,nameForXFlipped)

main()
