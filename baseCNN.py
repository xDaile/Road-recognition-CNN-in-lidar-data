#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import numpy
import getFiles
import parameters
import torch
import cv2
import Model
import sys
#import time
#from notify_run import Notify
import accuracyCalc


#training can be stopped by "touch stop" in current dir

#notifying own smartphone with this, see https://notify.run/c/2sgVnBxNtkkPi2oc
#notify = Notify()

criterion = torch.nn.CrossEntropyLoss(ignore_index=3)

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
        y = torch.load(self.GTDict[key])
        X=X.to(device)
        y=y.to(device)
        return X, y


#how often will be validation done - to avoid overfiting
view_step=10

#parametres for dataloaders
params = {"train":{
            'shuffle': True,
            #'batch_size': 64,
            'num_workers': 0} ,
        "test":{
            'shuffle': True,
            #'batch_size': 64,
            'num_workers': 0}}

#loading data - see dataset how this is used
tensors=getFiles.loadListOfTensors()
groundTruth=getFiles.getListOfGroundTruthFiles()
listIDs=getFiles.getListOfIDs()

#cuda device switch to nvidia
def get_device():
    if torch.cuda.is_available():
        global device
        device = torch.device('cuda:0')
        print("Device changed to: "+ torch.cuda.get_device_name(0))
    else:
        print("Device was not changed to gtx 960m")
        device = torch.device('cpu') # don't have GPU

#validation test,
def test(model, data_loader):
    #setting eval mode for not using dropout, and other things that help learning but not validation
    model=model.eval()
    model.to(device)
    loss_sum=0
    accuracy_sum=0
    iterations=0
    for inputForNetwork,outputFromNetwork in data_loader:
        result=model(inputForNetwork)
        loss=criterion(result,outputFromNetwork)
        loss_sum=loss_sum+loss.item()
        accuracy=accuracyCalc.accuracy(outputFromNetwork,result,device)
        accuracy_sum=accuracy_sum+accuracy
        iterations+=1
        break
    model=model.train()
    return loss_sum/iterations , accuracy_sum/iterations

get_device()

#initialization of dataloaders
training_set = Dataset(listIDs['train'],tensors['train'],groundTruth['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params['train'],batch_size=1)

validation_set = Dataset(listIDs['test'],tensors['test'],groundTruth['test'])
validation_generator = torch.utils.data.DataLoader(training_set, **params['test'])
iteration=1
#model needs to be created too if it will be loaded
model= Model.Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
if(os.path.exists(parameters.modelSavedFile)):
    print("Model will be loaded from saved state")
    checkpoint=torch.load(parameters.modelSavedFile)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    iteration=checkpoint['iteration']
    loss=checkpoint['loss']
    model.eval()
    get_device()
    model.to(device)
else:
    print("model not found, starting from scratch")

#this serves for stop training when needed
continueTraining=True
loss_sum=0
accuracy_sum=0



#training
while(continueTraining):
    #startTime=time.time()
    model.train()
    loss_sum=0
    accuracy_sum=0
    model.to(device)
    for inputForNetwork,outputFromNetwork in training_generator:
        result=model(inputForNetwork)
        loss = criterion(result,outputFromNetwork)
        optimizer.zero_grad()#see doc
        loss.backward()#see doc
        optimizer.step()#see doc
        loss_sum=loss_sum+loss.item()
        #print(time.timeit(accuracyCalc(outputFromNetwork,result),1))
        accuracy=accuracyCalc.accuracy(outputFromNetwork,result,device)
        accuracy_sum=accuracy_sum+accuracy
        break

    if(iteration%view_step==0):
        #validation
        test_loss, test_accuracy=test(model,validation_generator)

        #message for sent to notify mine smartphone
        message="Iteration:" + str(iteration) + "\nLoss:" + str(loss_sum/view_step) + "\nAccuracy:" + str(accuracy_sum/view_step) + "\nTestLoss:" + str(test_loss) + "\nTestAccuracy:" + str(test_accuracy)

        loss_sum=0
        accuracy_sum=0
        test_loss_sum=0
        test_accuracy_sum=0
        #happens that sending notify cannot be done, then it fails whole
        try:
            #notify.send(message)
            print(message)
        except:
            print("failed to send")

        #training can be stopped by "touch stop"
        if(os.path.exists("./stop")):
                print("saving model params")
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, parameters.modelSavedFile)
                continueTraining=False
    iteration=iteration+1
#    end=time.time()
#    print("Elapsed Time in 1 Epoch:",end-startTime)
