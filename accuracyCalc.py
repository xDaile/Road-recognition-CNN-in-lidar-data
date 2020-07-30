#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
   Author: Michal Zelenak
   BUT Faculty of Information Technology
   This is code written for the bachelor thesis
   Project: Object Detection in the Laser Scans Using Convolutional Neural Networks
"""
import torch

#function which count the accuracy and F-Measure of the predicted result
def accuracy(prediction, expectedResult,cuda0):
    expectedResult=expectedResult.float()

    prediction=prediction[0]
    road=prediction[0]

    #not road class
    notRoad=prediction[1]
    i=2
    while(i<len(prediction)):
        #add other classes to not road, because those are the classes where road is not, or we can not for sure tell
        notRoad+=prediction[i]
        i+=1

    road=road.to(device=cuda0)
    notRoad=notRoad.to(device=cuda0)
    zeros=torch.zeros(400,200).float()
    zeros=zeros.to(device=cuda0)
    ones=torch.ones(400,200).float()
    ones=ones.to(device=cuda0)

    #create tensor with only ones and zeros
    prediction=torch.where(road>=notRoad,zeros,ones)

    expectedResult=expectedResult.to(device=cuda0)

    class0Prediction=torch.where(prediction==0,ones,zeros)
    class0NeqPrediction=torch.where(prediction!=0,ones,zeros)
    class0Truth=torch.where(expectedResult==0,ones,zeros)
    class0NeqTruth=torch.where(expectedResult!=0,ones,zeros)
    TP=torch.mul(class0Prediction,class0Truth).sum()
    TN=torch.mul(class0NeqPrediction,class0NeqTruth).sum()
    FP=torch.mul(class0Prediction,class0NeqTruth).sum()
    FN=torch.mul(class0NeqPrediction,class0Truth).sum()
    #print(TP.item(),TN.item(),FP.item(),FN.item())
    try:
        precision=TP.item()/(TP.item()+FP.item())
        recall=TP.item()/(TP.item()+FN.item())
        maxF=2*((precision*recall)/(precision+recall))
        accuracy=(TP.item()+TN.item())/(TP.item()+TN.item()+FP.item()+FN.item())
    except:
        maxF= 0
        accuracy=0
    return accuracy,maxF
