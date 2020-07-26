#!/usr/bin/env python3
import numpy
import getFileLists
import torch
import matplotlib.pyplot as plt

zeros=torch.zeros(400,200).float()
ones=torch.ones(400,200).float()
groundTruthFilesList=getFileLists.getListOfGroundTruthFiles()
listIDs=getFileLists.getListOfIDs()
groundTruthSum=torch.zeros(400,200).float()
i=0
for file in listIDs["train"]:
#    print(groundTruthFilesList["train"][file])
    gt=torch.load(groundTruthFilesList["train"][file])
    groundTruthSum=torch.add(groundTruthSum,gt)
    i+=1
    print(i," from ", len(listIDs["train"]))
gtMean=torch.where(groundTruthSum>1,ones,groundTruthSum)
gtMean=torch.div(gtMean,len(listIDs))

universalResult=torch.where(gtMean<0.7,zeros,ones)
universalResultTreshold=universalResult.sum()/80000
print(universalResultTreshold)
torch.save(universalResult,"universalResultForRoad")
fig = plt.figure(figsize=(6, 3.2))
plt.imshow(universalResult,label="Trénovacia sada")
plt.show()
