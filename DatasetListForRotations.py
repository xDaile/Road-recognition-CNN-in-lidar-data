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

import os
import parameters

class DatasetList():

    def __init__(self):
       'Initialization'
       self.list_IDs = self.getListOfIDs()
       self.pclDict=self.getDictOfPclFiles()
       self.GTDict=self.getDictOfGT()
       self.position=0
       self.lenght=self.countOfFiles()
       self.itemsList=[]
       while(self.position<self.lenght):
           self.itemsList.append(self.getNextItem())
           self.position=self.position+1

    def getListOfIDs(self):
      place=parameters.pclFiles
      filesList=os.listdir(place)
      List=[]
      for item in filesList:
            fullName=place+item
            if(os.path.isfile(fullName)):
                key=item[:-4]
                List.append(key)
      return List

    def getDictOfPclFiles(self):
      place=parameters.pclFiles
      pclFilesList=os.listdir(place)
      pclDict={}
      for item in pclFilesList:
        key=item[:-4]
        fullName=place+item
        if(os.path.isfile(fullName)):
            pclDict.update({key:fullName})
      return pclDict

    def getDictOfGT(self):
      place=parameters.groundTruthImages
      GTList=os.listdir(place)
      GTDict={}
      for item in GTList:
          key=item[:-7]
          fullName=place+item
          GTDict.update({key:fullName})
      return GTDict


    def countOfFiles(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def getNextItem(self):
        'Generates one sample of data'
        # Select sample
        key = self.list_IDs[self.position]
        # Load data and get label
        X = self.pclDict[key]#HERE I ENDED
        y = self.GTDict[key]

        return X, y
