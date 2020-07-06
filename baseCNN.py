#!/usr/bin/env python3
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy
import getFiles
import parameters
import torch
import Model
import sys
#import time
from notify_run import Notify
import accuracyCalc
import subprocess

#cuda device switch to nvidia
def get_device():
    if torch.cuda.is_available():
        global device
        device = torch.device('cuda:0')
        print("Device changed to: "+ torch.cuda.get_device_name(0))
    else:
        print("Device was not changed to gtx 960m")
        device = torch.device('cpu') # don't have GPU
    #device = torch.device('cpu') # don't have GPU

#set output from network to 3 or 2?
get_device()
print("HERE")
print(torch.cuda.is_available())
#training can be stopped by "touch stop" in current dir

#notifying own smartphone with this, see https://notify.run/c/2sgVnBxNtkkPi2oc
notify = Notify()
volatile=True
ignore=torch.tensor([1,1,0]).to(device).float() #ignoring class 2 while computing loss
#ignore=ignore.to(device)
#ignore=ignore.float()
criterion = torch.nn.CrossEntropyLoss(reduction='sum',weight=ignore)


#how often will be validation done - to avoid overfiting


#parametres for dataloaders
params = {"train":{
            'shuffle': True,
            #'batch_size': 14,
            'num_workers': 0} ,
        "test":{
            'shuffle': True,
            #'batch_size': 64,
            'num_workers': 0}}


#criterion = torch.nn.BCELoss()
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
        #print("working with:",key)
        # Load data and get label
        X = torch.load(self.tensorDict[key])#HERE I ENDED

        y = torch.load(self.GTDict[key])
        X=X.to(device)
        y=y.to(device)
        return X, y




#loading data - see dataset how this is used
tensors=getFiles.loadListOfTensors()
groundTruth=getFiles.getListOfGroundTruthFiles()
listIDs=getFiles.getListOfIDs()


#validation test,
def test(model, data_loader):
    #setting eval mode for not using dropout, and other things that help learning but not validation
    model=model.eval()
    model.to(device)
    loss_sum=0
    accuracy_sum=0
    iterations=0
    maxF_sum=0
    for inputForNetwork,outputFromNetwork in data_loader:
        result=model(inputForNetwork)
        loss=criterion(result,outputFromNetwork)
        loss_sum=loss_sum+loss.item()
        max_f,accuracy=accuracyCalc.accuracy(outputFromNetwork,result,device)
        accuracy_sum=accuracy_sum+accuracy
        maxF_sum=maxF_sum+max_f
        iterations+=1
        #break
    model=model.train()
    return loss_sum/iterations , accuracy_sum/iterations,maxF_sum/iterations



#initialization of dataloaders
#print(groundTruth)
training_set = Dataset(listIDs['train'],tensors['train'],groundTruth['train'])
training_generator = torch.utils.data.DataLoader(training_set, **params['train'])

validation_set = Dataset(listIDs['test'],tensors['test'],groundTruth['test'])
validation_generator = torch.utils.data.DataLoader(training_set, **params['test'])


continueTraining=True
loss_sum=0
accuracy_sum=0
maxF_sum=0
iteration=0
learning_rate=0.001

#model needs to be created too if it will be loaded
model= Model.Net()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
    model.eval()
    get_device()
    model.to(device)
else:
    print("model not found, starting from scratch")

#this serves for stop training when needed


def sendMessage(message):
    try:
        notify.send(message)
    except:
        print(message)

def saveModel(model,iteration,optimizer):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, parameters.modelSavedFile)

def saveMaxACCModel(mode,iteration,optimizer,acc):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy':acc,
        }, "MaxACCModel.tar")

def saveModelByIterations(mode,iteration,optimizer):
    saveModel(model,iteration,optimizer)
    subprocess.call("./sendModel.sh", shell=True)
    #now send model to cloud
    return 0

def saveModelByTouchStop(model,iteration,optimizer):
    if(os.path.exists("./stop")):
            print("saving model params")
            saveModel(model,iteration,optimizer)
            exit()
            return False
    return True

#iteration=1
view_step=2000
MaxACC=0
#      0 1 2 3 4 5 6 7 8 9
#maxes=[0,0,0,0,0,0,0,0,0,0]
#lenMaxes=len(maxes)
epochWithoutChange=0
#training
changedMax=False
while(continueTraining):
    iteration=iteration+1

    model.train()
    model.to(device)
    numOfSamples=0
    for inputForNetwork,outputFromNetwork in training_generator:


        #for some reason, data loader is adding one more dimension - because batch
        numOfSamples=numOfSamples+1
        result=model(inputForNetwork)

        loss = criterion(result,outputFromNetwork)

        optimizer.zero_grad()#see doc
        loss.backward() #see doc
        optimizer.step()#see doc
        loss_sum=loss_sum+loss.item()
        #print(time.timeit(accuracyCalc(outputFromNetwork,result),1))
        maxF,accuracy=accuracyCalc.accuracy(outputFromNetwork,result,device)
        accuracy_sum=accuracy_sum+accuracy
        maxF_sum=maxF_sum+maxF
        #break
        if(epochWithoutChange>2):
            view_step=int(view_step/2)
            epochWithoutChange=0
            learning_rate=learning_rate/2
            message="learning rate changed to:"+str(learning_rate)
            sendMessage(message)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        if(numOfSamples%view_step==0):
            #validation
            test_loss, test_accuracy,test_maxF=test(model,validation_generator)

            #message for sent to notify mine smartphone
            message=" MaxAccuracy: "+str(MaxACC) + "\tEpoch:"+str(iteration)+"\tLoss:" + "{:.0f}".format(loss_sum/(view_step)) + "\tAccuracy:" + "{:.2f}".format(accuracy_sum/(view_step))+"\tMaxF: "+"{:.2f}".format(maxF_sum/view_step) + "\tTestLoss:" + "{:.0f}".format(test_loss) + "\tTestAccuracy:" + "{:.2f}".format(test_accuracy)+"\tTestMaxF:"+"{:.2f}".format(test_maxF)
            measureACC=test_accuracy
            #print(message)
            if(measureACC>(MaxACC)):
                MaxACC=measureACC
                saveMaxACCModel(model,iteration,optimizer,MaxACC)
                changedMax=True
            loss_sum=0
            accuracy_sum=0
            maxF_sum=0


            #happens that sending notify cannot be done, then it fails whole
            sendMessage(message)

            #training can be stopped by "touch stop"
            continueTraining=saveModelByTouchStop(model,iteration,optimizer)
    if(changedMax==False):
        epochWithoutChange=epochWithoutChange+1
    else:
        changedMax=False

    #if(iteration%save_step==0):
    #    saveModelByIterations(model,iteration,optimizer)
