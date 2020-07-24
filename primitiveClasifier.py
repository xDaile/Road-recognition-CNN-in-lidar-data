#!/usr/bin/env python3
import numpy
import getFileLists
import torch
import matplotlib.pyplot as plt

groundTruthFilesList=getFileLists.getListOfGroundTruthFiles()
listIDs=getFileLists.getListOfIDs()
groundTruthSum=[]
i=0
for file in listIDs["train"]:
#    print(groundTruthFilesList["train"][file])
    gt=torch.load(groundTruthFilesList["train"][file])
    groundTruthSum+=gt
    i+=1
    print(i," from ", len(listIDs["train"]))


gtMean=groundTruthSum/len(groundTruth)
zeros=torch.zeros(400,200).float()
ones=torch.ones(400,200).float()
universalResult=torch.where(gtMean<0.5,zeros,ones)
torch.save(universalResult,"universalResultForRoad")
fig = plt.figure(figsize=(6, 3.2))
plt.imshow(universalResult,label="Trénovacia sada")
plt.show()
