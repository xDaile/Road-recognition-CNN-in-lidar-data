#!/usr/bin/env python3
import os
import cv2
import parameters
import numpy

#load ground truth images from png to numpy
place=parameters.originalGT
GTList=os.listdir(place)
for gtItem in GTList:
    name=place+gtItem
    if(os.path.isfile(name)):
        fullName=place+gtItem
        img=cv2.imread(fullName)
        toSave=numpy.array(img)
        toSave=numpy.subtract(toSave,1)
        i=0
        newGT=[[0 for i in range(200)] for j in range(400)]
        while(i<400):
            j=0
            while(j<200):
                newGT[i][j]=toSave[i][j][0]
                j+=1
            i+=1
        newName=fullName.replace("groundTruth","GroundTruth")
        newName=newName[:-4]
        print(newName)
        numpy.save(newName,numpy.array(newGT),allow_pickle=True)
