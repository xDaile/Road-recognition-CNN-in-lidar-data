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
    gtMean=torch.where(gt>1,ones,zeros)
    groundTruthSum=torch.add(gtMean,gt)
    i+=1
    print(i," from ", len(listIDs["train"]))

gtMean=torch.div(groundTruthSum,len(listIDs))

universalResultTreshold=gtMean.sum()/80000
print(universalResult.shape)
treshold=universalResultTreshold.item()
print(treshold)
universalResult=torch.where(gtMean<treshold,zeros,ones)

torch.save(universalResult,"universalResultForRoad")
fig = plt.figure(figsize=(6, 3.2))
plt.imshow(universalResult,label="Trénovacia sada")
plt.show()
