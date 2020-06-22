#!/usr/bin/env python3
import torch
import torchfile
import sys
import numpy as np

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

if (len(sys.argv)!=2):
    print("usage: ./TensorFlipper nameOfTensor\n Tensor at the output will be flipped by x, by x and y, and by y, for each flipp will be created new file")
    exit(1)

nameOfFile=sys.argv[1]

orig=torch.load(nameOfFile)

#newNumber=addNumberToNumberAtName(nameOfFile,100)


flippedByX=flipByX(orig)
#flippedByY=flipByY(orig)
#flippedByXY=flipByXY(orig)
nameForXFlipped=addNumberToNumberAtName(nameOfFile,100)
#nameForYFlipped=addNumberToNumberAtName(nameOfFile,200)
#nameForXYFlipped=addNumberToNumberAtName(nameOfFile,300)

torch.save(flippedByX,nameForXFlipped)
#torch.save(flippedByY,nameForYFlipped)
#torch.save(flippedByXY,nameForXYFlipped)
