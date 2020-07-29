#!/usr/bin/env python3
from __future__ import print_function
from notify_run import Notify
import matplotlib.pyplot as plt
import os
import subprocess
import sys
import numpy
import torch
import getFileLists
import parameters
import Model
import accuracyCalc
import toolsForNetwork

#how often will be neural network tested
view_step=1000
#how fast will neural network learn
learning_rate=0.0001
cuda0=toolsForNetwork.get_device()
#notifying own smartphone with this, see https://notify.run/c/2sgVnBxNtkkPi2oc
notify = Notify()
volatile=True

#weigh parameter
ignore=torch.tensor([1,1,1,1]).float() #ignoring class 2 while computing loss
ignore=ignore.to(device=cuda0)
criterion = torch.nn.CrossEntropyLoss(reduction='mean',weight=ignore)
numberOfTrainingCycles=21
#how fastt wil learning rate fall - with this number will be learning rate divided every time two epoch will not have better results
learningRateDiv=5
#structure for saving results
results={"train":                           \
            {"Loss":[],                     \
            "Accuracy-precise":[],          \
            "MaxF-precise":[],              \
            "Accuracy":[],                  \
            "MaxF":[]},                     \
        "test":                             \
            {"Loss":[],                     \
            "Accuracy-precise":[],          \
            "MaxF-precise":[],              \
            "Accuracy":[],                  \
            "MaxF":[]},                      \
        "epoch":[]}

#parametres for dataloaders
params = {"train":{
            'shuffle': True,
            #'batch_size': 14,
            'num_workers': 0} ,
        "test":{
            'shuffle': True,
            #'batch_size': 64,
            'num_workers': 0}}

#training can be stopped by "touch stop" in current dir
def saveModelByTouchStop(model,iteration,optimizer,acc):
    if(os.path.exists("./stop")):
            print("saving model params")
            saveModel(model,iteration,optimizer,acc)
            saveResultsOnDisk()
            exit()
            return False
    return True




def saveResults(trainLoss,trainACCPrecise,trainMaxFPrecise,trainAcc,trainMaxf,testLoss,testACCPrecise,testMaxFPrecise,testAcc,testMaxf,iteration):
    results["train"]["Loss"].append(trainLoss)
    results["train"]["Accuracy-precise"].append(trainACCPrecise)
    results["train"]["MaxF-precise"].append(trainMaxFPrecise)
    results["train"]["Accuracy"].append(trainAcc)
    results["train"]["MaxF"].append(trainMaxf)
    results["test"]["Loss"].append(testLoss)
    results["test"]["Accuracy-precise"].append(testACCPrecise)
    results["test"]["MaxF-precise"].append(testMaxFPrecise)
    results["test"]["Accuracy"].append(testAcc)
    results["test"]["MaxF"].append(testMaxf)
    results["epoch"].append(iteration)




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

        # Load data and get label
        X = torch.load(self.tensorDict[key])#HERE I ENDED

        y = torch.load(self.GTDict[key])
        X=X.to(device=cuda0)
        y=y.to(device=cuda0)
        return X, y,key

#function that tests acutal neural network performance
def test(model, data_loader):
    #setting eval mode for not using dropout, and other things that help learning but not validation
    model=model.eval()
    model.to(device=cuda0)
    loss_sum=0
    accuracy_sum=0
    iterations=0
    maxF_sum=0
    maxF_Precise=0
    acc_Precise=0
    originalSamplesCounter=0
    samplesCounter=0
    var_sum=0
    for inputForNetwork,expectedOutputFromNetwork,key in data_loader:
        outputFromNetwork=model(inputForNetwork)
        loss=criterion(outputFromNetwork,expectedOutputFromNetwork)
        loss_sum=loss_sum+loss.item()
        accuracy,max_f=accuracyCalc.accuracy(outputFromNetwork,expectedOutputFromNetwork,cuda0)

        #count only original dataset results
        if(key[0][-3]=='0' and key[0][-4]=='0'):
            maxF_Precise+=max_f
            acc_Precise+=accuracy
            originalSamplesCounter+=1
        accuracy_sum=accuracy_sum+accuracy
        maxF_sum=maxF_sum+max_f
        samplesCounter+=1
    model=model.train()
    return loss_sum/samplesCounter , accuracy_sum/samplesCounter,maxF_sum/samplesCounter,acc_Precise/originalSamplesCounter,maxF_Precise/originalSamplesCounter

def sendMessage(message):
    try:
        notify.send(message)
    except:
        print(message)

def saveModel(model,iteration,optimizer,acc):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy':acc,
        }, parameters.modelSavedFile)
    saveResultsOnDisk()

def saveModelByIterations(model,iteration,optimizer,acc):
    print("saving model params")
    saveModel(model,iteration,optimizer,acc)
    saveResultsOnDisk()
    exit(0)

def saveResultsOnDisk():
    f = open("results.txt","w")
    f.write( str(results) )
    f.close()

def main():
    global learning_rate
    global view_step
    global cuda0
    global notify
    global criterion
    global numberOfTrainingCycles
    global learningRateDiv
    global results
    global params
    #loading data - see dataset how this is used
    tensors=getFileLists.loadListOfTensors()
    groundTruth=getFileLists.getDictOfGroundTruthFiles()
    listIDs=getFileLists.getListOfIDs()

    #initialization of dataloaders
    training_set = Dataset(listIDs['train'],tensors['train'],groundTruth['train'])
    training_generator = torch.utils.data.DataLoader(training_set, **params['train'])

    validation_set = Dataset(listIDs['test'],tensors['test'],groundTruth['test'])
    validation_generator = torch.utils.data.DataLoader(validation_set, **params['test'])

    continueTraining=True
    loss_sum=0
    accuracy_sum=0
    maxF_sum=0
    iteration=0

    #model needs to be created too if it will be loaded
    model= Model.Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #try to load saved model, if some exists
    if(os.path.exists(parameters.modelSavedFile)):
        print("Model will be loaded from saved state")
        checkpoint=torch.load(parameters.modelSavedFile)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device=cuda0)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        iteration=checkpoint['iteration']
        model.eval()
        toolsForNetwork.get_device()
        model.to(device=cuda0)
    else:
        print("model not found, starting from scratch")

    #parameter which best f-measure
    bestFM=0
    #counter of training cycles which were without change
    epochWithoutChange=0

    #training of the neural network
    while(continueTraining):
        iteration=iteration+1

        #parameter if the maximum accuracy was changed
        changedMaxFM=False

        #convert model to trainable state
        model.train()

        #move model data to gpu
        model.to(device=cuda0)

        #sum of the accuracy on the training cycle
        accuracy_sum=0

        #counter of the original datas
        withoutACCmiss=0

        #F-Measure sum
        maxF_sum=0

        #value of the F-measure counted only on original dataset - rotated by 0°
        maxF_Precise=0

        #value of the accuracy counted only on original Dataset - rotated by 0°
        acc_Precise=0

        #number of trained samples
        numOfSamples=0

        #Flag if the trainable data are from original Dataset - rotated by 0°
        origSample=False

        #going thtough all the dataset
        for inputForNetwork,outputFromNetwork,key in training_generator:

            #check if the current sample is  rotated by 0°
            if(key[0][-3]=='0' and key[0][-4]=='0'):
                origSample=True

            numOfSamples=numOfSamples+1

            #get the output from the neural network
            result=model(inputForNetwork)

            #count loss value
            loss = criterion(result,outputFromNetwork)

            #use optimizer
            optimizer.zero_grad()#see doc

            #count new parameters of the neural network
            loss.backward() #see doc
            optimizer.step()#see doc

            #count loss throught all the dataset
            loss_sum=loss_sum+loss.item()

            #get the accuracy and F-Measure of the current prediction from the network and ground truth
            accuracy,maxF=accuracyCalc.accuracy(result,outputFromNetwork,cuda0)
            if(origSample):
                withoutACCmiss+=1
                maxF_Precise+=maxF
                acc_Precise+=accuracy
                origSample=False
            accuracy_sum+=accuracy
            maxF_sum+=maxF

            #if it is time to check the model performance
            if(numOfSamples%view_step==0):

                test_loss, test_accuracy,test_MaxF,test_acc_precise,test_MaxF_precise=test(model,validation_generator)

                #message for sent to notifing own smartphone
                message="Epoch:"                            + str(iteration)                                \
                        + " \tMaximal f-measure: "            + "{:.2f}".format(bestFM*100)                               \
                        + " \tTRAIN  Loss Value:"             + "{:.2f}".format(loss_sum/(view_step))         \
                        + " \tTRAIN  Accuracy - precise:"     + "{:.2f}".format((maxF_Precise*100)/withoutACCmiss)  \
                        + " \tTRAIN  F-Measure - precise:"         + "{:.2f}".format((acc_Precise*100)/withoutACCmiss)   \
                        + " \tTRAIN  Accuracy:"               + "{:.2f}".format((accuracy_sum*100)/(view_step))     \
                        + " \tTRAIN  F-Measure: "                  + "{:.2f}".format((maxF_sum*100)/view_step)           \
                        + " \tTEST  Loss Value:"              + "{:.2f}".format(test_loss)                    \
                        + " \tTEST  Accuracy - precise:"      + "{:.2f}".format(test_acc_precise*100)             \
                        + " \tTEST  F-Measure - precise:"          + "{:.2f}".format(test_MaxF_precise*100)            \
                        + " \tTEST  Accuracy:"                + "{:.2f}".format(test_accuracy*100)                \
                        + " \tTEST  F-Measure:"                    + "{:.2f}".format(test_MaxF*100)

                saveResults(loss_sum/view_step,             \
                            acc_Precise/withoutACCmiss,     \
                            maxF_Precise/withoutACCmiss,    \
                            accuracy_sum/view_step,         \
                            maxF_sum/view_step,             \
                            test_loss,                      \
                            test_acc_precise,               \
                            test_MaxF_precise,              \
                            test_accuracy,                  \
                            test_MaxF,                      \
                            iteration)

                if(test_MaxF>(bestFM)):
                    bestFM=test_MaxF
                    print(bestFM)
                    changedMaxFM=True
                loss_sum=0
                accuracy_sum=0
                maxF_sum=0
                var_sum=0
                sendMessage(message)

                #training can be stopped by "touch stop" - now it is time to check if the file called 'stop' exists
                continueTraining=saveModelByTouchStop(model,iteration,optimizer,bestFM)

        #if there were two training cycles without change, decay learning rate
        if(epochWithoutChange==2):
            epochWithoutChange=0
            learning_rate=learning_rate/learningRateDiv
            message="Learning rate changed to:"+str(learning_rate)
            sendMessage(message)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        #check if the training cycle get the better f-measure than previous
        if(changedMaxFM==False):
            epochWithoutChange=epochWithoutChange+1
        #stop training by number of iterations
        if(iteration==numberOfTrainingCycles):
            saveModelByIterations(model,iteration,optimizer,bestFM)
            exit(0)

if __name__ == "__main__":
    main()
