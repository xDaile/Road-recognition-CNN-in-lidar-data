#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import numpy
import getFiles
import parameters
import torch
import torchfile
import cv2
import Model
import sys
from notify_run import Notify
from sklearn.metrics import confusion_matrix

notify = Notify()


def accuracy(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    print(confusion_vector)
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()
    precision=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)
    maxF=2*((precision*recall)/(precision+recall))
    return maxF

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

from notify_run import Notify
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
if(os.path.exists("./model.pth")):
    print("Model will be loaded from saved state")
    model.load_state_dict(torch.load("./model.pth"))
    model.eval()
else:
    print("model not found, starting from scratch")


criterion = torch.nn.CrossEntropyLoss(ignore_index=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
view_step=10
continueTraining=True
loss_acc=0
accuracy_acc=0
iteration=0
model.train()

def test(model, data_loader):
    model=model.eval()
    loss_acc=0
    accuracy_acc=0
    counter=0
    for inputForNetwork,outputFromNetwork in data_loader:
        result=model(inputForNetwork)
        loss=criterion(result,outputFromNetwork)
        loss_acc=loss_acc+loss.item()
        accuracy=accuracy(outputFromNetwork,result)
        accuracy_acc=accuracy_acc+accuracy
        counter+=1
        #break
    model=model.train()
    return loss_acc/counter , accuracy_acc/counter


i=0
while(continueTraining):

    lenghtOfTrainingInputs=len(training_generator)
    model.to(device)
    for inputForNetwork,outputFromNetwork in training_generator:
        result=model(inputForNetwork)
        loss = criterion(result,outputFromNetwork)
        optimizer.zero_grad()#see doc
        loss.backward()#see doc
        optimizer.step()#see doc
        loss_acc=loss_acc+loss.item()
        accuracy=accuracy(outputFromNetwork,result)
        accuracy_acc=accuracy_acc+accuracy
        #break

    if(i%view_step==0):
        test_loss_acc, test_accuracy_acc=test(model,validation_generator)
        message="Iteration:" + str(i) + "\nLoss:" + str(loss_acc/view_step) + "\nAccuracy:" + str(accuracy_acc/view_step) + "\nTestLoss:" + str(test_loss_acc) + "\nTestAccuracy_acc:" + str(test_accuracy_acc)
        loss_acc=0
        accuracy_acc=0
        test_loss_acc=0
        test_accuracy_acc=0
        try:
            notify.send(message)
        except:
            print("failed to send")

        if(os.path.exists("./stop")):
                print("saving model params")
                model.eval()
                torch.save(model.state_dict(), "./model.pth")
                continueTraining=False
    i=i+1
