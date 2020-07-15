#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import torch
import Model

model=Model.Net()
fileWithModel="./trainingDone/1/MaxACCModel.tar"
checkpoint=torch.load(fileWithModel)
#iteration=checkpoint['iteration']

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

testSubject=torch.load("./trainingDone/um_010085")
testSubject=torch.stack([testSubject])
outputFromModel=model(testSubject)
numpyAr=outputFromModel.detach().numpy()
numpyAr=numpyAr[0]
out = [[0 for i in range(200)] for j in range(400)]
i=0
while(i<400):
    j=0
    while(j<200):
        #if(i>300):
        if(numpyAr[0][i][j]>numpyAr[1][i][j]):
            out[i][j]=1#,numpyAr[2][i][j])
        else:
            out[i][j]=0
        #else:
        #    out=max(numpAr[0][i][j],numpAr[1][i][j])
        j=j+1
    i=i+1

fig = plt.figure(figsize=(6, 3.2))
plt.imshow(out,label="Trénovacia sada")
plt.show()
#print(numpyAr.shape)
