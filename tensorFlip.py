#!/usr/bin/env python3
import torch
import torchfile
import sys
import numpy as np

#Works perfect
def flipByYStackedTensor(tensor):
    flipped=torch.flip(tensor,(1,0))
    return torch.stack([flipped[5],flipped[4],flipped[3],flipped[2],flipped[1],flipped[0]])

def flipByXY(tensor):
    rotated=torch.rot90(tensor,1,[1,0])
    rotatedFlipped=torch.flip(rotated,[0,1])
    flippedByXY=torch.rot90(rotatedFlipped,1 ,[0,1])
    return flippedByXY


#works perfect
def flipByXYStackedTensor(tensor):
    i=0
    new=[]
    while(i<6):
        new.append(flipByXY(tensor[i]))
        i=i+1
    end=torch.stack(new)
    return end

def flipByXStackedTensor(tensor):#HERE WORKING !!!!
    return flipByYStackedTensor(flipByXYStackedTensor(tensor))

def addNumberToNumberAtName(name,number):
    possibleNumber=[100,200,300]
    if(number in possibleNumber):
        fileNumber=int(name[-3:])
        if(fileNumber>99):
            return 0
        number=fileNumber+number
        file=name[:-3]
        return (file+str(number))
    else:
        print("that number cannot be assigned to name of the file")
        exit(1)

if (len(sys.argv)!=2):
    print("usage: ./TensorFlipper nameOfTensor\n Tensor at the output will be flipped by x, by x and y, and by y, for each flipp will be created new file")
    exit(1)

nameOfFile=sys.argv[1]
print(nameOfFile)
orig=torch.load(nameOfFile)

flippedByXY=flipByXYStackedTensor(orig)
flippedByY=flipByYStackedTensor(orig)
flippedByX=flipByXStackedTensor(orig)

nameForXFlipped=addNumberToNumberAtName(nameOfFile,100)
nameForYFlipped=addNumberToNumberAtName(nameOfFile,200)
nameForXYFlipped=addNumberToNumberAtName(nameOfFile,300)
#print(nameForXFlipped)
#print(orig[4])
#print(flippedByXY[4])

torch.save(flippedByX,nameForXFlipped)
torch.save(flippedByY,nameForYFlipped)
torch.save(flippedByXY,nameForXYFlipped)
