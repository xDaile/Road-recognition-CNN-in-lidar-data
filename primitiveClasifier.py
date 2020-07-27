#!/usr/bin/env python3
import numpy
import getFileLists
import torch
import matplotlib.pyplot as plt

zeros=torch.zeros(400,200).float()
ones=torch.ones(400,200).float()
groundTruthFilesList=getFileLists.getDictOfGroundTruthFiles()
listIDs=getFileLists.getListOfIDs()
gtClass0Sum=torch.zeros(400,200).float()
gtClass1Sum=torch.zeros(400,200).float()
gtClass2Sum=torch.zeros(400,200).float()
gtClass3Sum=torch.zeros(400,200).float()
newClasifier=torch.zeros(400,200).float()



i=0
#going throught dataset
for file in listIDs["train"]:

    #if 'training' only on original dataset uncoment next line
    #if(file[-3]=='0' and file[-4]=='0'and file[-6]=='0' ):
        gt=torch.load(groundTruthFilesList["train"][file])
        #get ground truth values from gt by classes
        gtClass0=torch.where(gt==0,ones,zeros)
        gtClass1=torch.where(gt==1,ones,zeros)
        gtClass2=torch.where(gt==2,ones,zeros)
        gtClass3=torch.where(gt==3,ones,zeros)

        #sum those ground truth values by classes
        gtClass0Sum=torch.add(gtClass0Sum,gtClass0)
        gtClass1Sum=torch.add(gtClass1Sum,gtClass1)
        gtClass2Sum=torch.add(gtClass2Sum,gtClass2)
        gtClass3Sum=torch.add(gtClass3Sum,gtClass3)
        i+=1
        print(i," from ", len(listIDs["train"]))

#now we have 4 tensors, each representing his own class
#tensor with dimensions [400]x[200] will be created, in each cell will be value for the class, which had most ocurencies at this place
i=0
while(i<400):
    j=0
    while(j<200):
        maxVal=max(gtClass0Sum[i][j],gtClass1Sum[i][j],gtClass2Sum[i][j],gtClass3Sum[i][j])
        if(maxVal==gtClass0Sum[i][j]):
            newClasifier[i][j]=0
        if(maxVal==gtClass1Sum[i][j]):
            newClasifier[i][j]=1
        if(maxVal==gtClass2Sum[i][j]):
            newClasifier[i][j]=1
        if(maxVal==gtClass3Sum[i][j]):
            newClasifier[i][j]=1
        j+=1
    i+=1

#save the result
torch.save(newClasifier,"universalResultForRoad")

#show the resutls
fig = plt.figure(figsize=(6, 3.2))
plt.imshow(newClasifier,label="Trénovacia sada")
plt.show()
