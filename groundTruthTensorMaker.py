#!/usr/bin/env python3

import os
import torch
import cv2

def transformGT(loadedGroundTruth):
    i=0
    j=0
    road=[]
    notRoad=[]
    third=[]
    for line in loadedGroundTruth:#400 times will do
        addRoadRow=[]
        addNotRoadRow=[]
        addThirdRow=[]
        for matrix in line:#200 times will do

            if(matrix[0]==1):
                    addRoadRow.append(1)
                    addNotRoadRow.append(0)
                    addThirdRow.append(0)
            else:
                if(matrix[0]==2):
                    addRoadRow.append(0)
                    addNotRoadRow.append(1)
                    addThirdRow.append(0)
                else:
                    addRoadRow.append(0)
                    addNotRoadRow.append(0)
                    addThirdRow.append(1)
            j=j+1
        road.append(addRoadRow)
        notRoad.append(addNotRoadRow)
        third.append(addThirdRow)
        addRoadRow=[]
        addNotRoadRow=[]
        addThirdRow=[]
        i=i+1
    final=[]
    final.extend([torch.tensor(road),torch.tensor(notRoad),torch.tensor(third)])
    return torch.stack(final)


gtTest=os.listdir("./groundTruth/test")
gtTrain=os.listdir("./groundTruth/train")
for file in gtTest:
    nameOfFileToOpen="./groundTruth/test/"+file
    y = cv2.imread(nameOfFileToOpen)
    y= transformGT(y)
    newName="./groundTruthTensors/test/"+file[:-4]
    print(newName)
    torch.save(y,newName)
    #print(s)

for file in gtTrain:
    nameOfFileToOpen="./groundTruth/train/"+file
    y = cv2.imread(nameOfFileToOpen)
    y= transformGT(y)
    newName="./groundTruthTensors/train/"+file[:-4]
    print(newName)
    torch.save(y,newName)
