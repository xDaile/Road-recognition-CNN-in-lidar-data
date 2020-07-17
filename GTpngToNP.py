#!/usr/bin/env python3
import os
import cv2
import parameters
import numpy

place="./groundTruth/"
GTList=os.listdir(place)
for gtItem in GTList:
    name=place+gtItem
    if(os.path.isfile(name)):
        fullName=place+gtItem
        img=cv2.imread(fullName)
        toSave=numpy.array(img)
        #print(toSave)
        toSave=numpy.subtract(toSave,1)
        #print(toSave)
        newName=fullName.replace("groundTruth","GroundTruth")
        newName=newName[:-4]
        print(newName)
        numpy.save(newName,toSave,allow_pickle=False)
    #print(toSave)
