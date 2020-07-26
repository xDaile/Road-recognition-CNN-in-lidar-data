#!/usr/bin/env python3
import numpy
import getFileLists
import torch
import matplotlib.pyplot as plt

zeros=torch.zeros(400,200).float()
ones=torch.ones(400,200).float()
groundTruthFilesList=getFileLists.getListOfGroundTruthFiles()
listIDs=getFileLists.getListOfIDs()
#groundTruthSum=torch.zeros(400,200).float()
gtclass0Sum=torch.zeros(400,200).float()
gtclass1Sum=torch.zeros(400,200).float()
gtclass2Sum=torch.zeros(400,200).float()
gtclass3Sum=torch.zeros(400,200).float()
newClasifier=torch.zeros(400,200).float()



i=0
for file in listIDs["train"]:
#    print(groundTruthFilesList["train"][file])

    #if(file[-3]=='0' and file[-4]=='0'and file[-6]=='0' ):

    gt=torch.load(groundTruthFilesList["train"][file])
    gtClass0=torch.where(gt==0,ones,zeros)
    gtClass1=torch.where(gt==1,ones,zeros)
    gtClass2=torch.where(gt==2,ones,zeros)
    gtClass3=torch.where(gt==3,ones,zeros)
    gtclass0Sum=torch.add(gtclass0Sum,gtClass0)
    gtclass1Sum=torch.add(gtclass1Sum,gtClass1)
    gtclass2Sum=torch.add(gtclass2Sum,gtClass2)
    gtclass3Sum=torch.add(gtclass3Sum,gtClass3)
    i+=1
    print(i," from ", len(listIDs["train"]))

i=0
while(i<400):
    j=0
    while(j<200):
        newClasifier[i][j]=max(gtClass0Sum[i][j],gtClass1Sum[i][j],gtClass2Sum[i][j],gtClass3Sum[i][j])
        j+=1
    i+-1
#gtMeanNew=torch.div(groundTruthSum,289)

#universalResultTreshold=gtMeanNew.sum()/80000
#treshold=universalResultTreshold.item()
#print(treshold)
#universalResult=torch.where(gtMean<treshold,zeros,ones)

torch.save(newClasifier,"universalResultForRoad")
fig = plt.figure(figsize=(6, 3.2))
plt.imshow(universalResult,label="Trénovacia sada")
plt.show()
