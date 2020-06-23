#!/usr/bin/env python3
import torch
import torchfile
import sys
import numpy as np
import os
import parameters

#works perfect
def flipByXY(tensor):
    return torch.flip(tensor,(0,1))

def flipByY(tensor):
    tensorInArray=tensor.numpy()
    i=0
    flipped= [[0] * 200] * 400
    while(i<400):
        flipped[399-i]=tensorInArray[i]
        i=i+1
    return torch.tensor(flipped)

def flipByX(tensor):
    return torch.stack([flipByXOneDimension(tensor[0]),flipByXOneDimension(tensor[1]),flipByXOneDimension(tensor[2])])

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

def main():
    gtTest=os.listdir(parameters.groundTruthTestTensorsFolder)
    gtTrain=os.listdir(parameters.groundTruthTrainTensorsFolder)

    for nameOfFile in gtTest:
        fullFileName=parameters.groundTruthTestTensorsFolder+nameOfFile

        orig=torch.load(fullFileName)
        flippedByX=flipByXOneDimension(orig)
        nameForXFlipped=addNumberToNumberAtName(fullFileName,100)
        print(nameForXFlipped)
        torch.save(flippedByX,nameForXFlipped)

    for nameOfFile in gtTrain:
        fullFileName=parameters.groundTruthTrainTensorsFolder+nameOfFile
        orig=torch.load(fullFileName)
        flippedByX=flipByXOneDimension(orig)
        nameForXFlipped=addNumberToNumberAtName(fullFileName,100)
        print(nameForXFlipped)
        torch.save(flippedByX,nameForXFlipped)



main()
