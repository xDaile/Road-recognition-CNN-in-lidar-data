#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import getFiles
import parameters
import torch
import torchfile
import cv2
import Model


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, tensorDict,GTDict):
        'Initialization'
        self.list_IDs = list_IDs
        self.tensorDict=tensorDict
        self.GTDict=GTDict
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        key = self.list_IDs[index]

        # Load data and get label
        X = torch.load(self.tensorDict[key])#HERE I ENDED
        y = cv2.imread(self.GTDict[key])
        #cv2.imshow("image",y)
        #cv2.waitKey(300)
        return X, y


def get_device():
    if torch.cuda.is_available():
        global device
        device = torch.device('cuda:0')

        print("Device changed to: "+ torch.cuda.get_device_name(0))
    else:
        print("Device was not changed to gtx 960m")
        device = torch.device('cpu') # don't have GPU

get_device()
tensors=getFiles.loadListOfTensors()
groundTruth=getFiles.getListOfGroundTruthFiles()
listIDs=getFiles.getListOfIDs()


params = {"train":{
            'shuffle': True,
            #'batch_size': 64,
            'num_workers': 0} ,
        "test":{
            'shuffle': True,
            #'batch_size': 64,
            'num_workers': 0}}

training_set = Dataset(listIDs['train'],tensors['train'],groundTruth['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params['train'],batch_size=1)

validation_set = Dataset(listIDs['test'],tensors['test'],groundTruth['test'])
validation_generator = torch.utils.data.DataLoader(training_set, **params['test'])

model= Model.Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#print(model)

stopTraining=True
#loop while not overfitted

while(stopTraining):
    for inputForNetwork,outputFromNetwork in training_generator:

        #print(inputForNetwork,outputFromNetwork)
        #print(inputForNetwork)
        #model.train()
        out=model.forward(inputForNetwork)
        print(out)
        stopTraining=False

        #inputForNetwork=inputForNetwork.to(device)
        #outputFromNetwork=outputFromNetwork.to(device)

#l=training_set.__getitem__(1)
#print(l)
