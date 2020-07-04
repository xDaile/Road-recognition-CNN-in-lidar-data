#!/usr/bin/env python3
#!/usr/bin/env python3
import torch
#import time

def get_device():
    if torch.cuda.is_available():
        global device
        device = torch.device('cuda:0')
        print("Device changed to: "+ torch.cuda.get_device_name(0))
    else:
        print("Device was not changed to gtx 960m")
        device = torch.device('cpu') # don't have GPU

def accuracy(truth, prediction,device):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
#third class is not computed into acc
    #start=time.time()
        #1 - road
        #2 - not road
        #3 - not used for compute TODO
    zeros=torch.zeros(1,400,200)
    zeros=zeros.to(device)

    ones=torch.ones(1,400,200)
    ones=ones.to(device)

    truth=truth.float()

#Split truth to 3 tensors for each class
    class0Truth=torch.where(truth==0,ones,zeros)
    class1Truth=torch.where(truth==1,ones,zeros)
    class2Truth=torch.where(truth==2,ones,zeros)
    class3Truth=torch.where(truth==3,ones,zeros)


# create the negation of the tensors created before this
    class0NeqTruth=torch.where(truth!=0,ones,zeros)
    class1NeqTruth=torch.where(truth!=1,ones,zeros)
    class2NeqTruth=torch.where(truth!=2,ones,zeros)
    class3NeqTruth=torch.where(truth!=3,ones,zeros)

#    for line int prediction:
#        for item in line:
    '''
    i=0
    predicted = [[0 for i in range(200)] for j in range(400)]
    while(i<400):
        j=0
        while(j<200):
            class0Value=prediction[0][0][i][j]
            class1Value=prediction[0][1][i][j]
            class2Value=prediction[0][2][i][j]
            maxV=max(class0Value,class1Value,class2Value)
            if(maxV==class0Value):
                predicted[i][j]=0
            if(maxV==class1Value):
                predicted[i][j]=1
            if(maxV==class2Value):
                predicted[i][j]=2
            j+=1
        i+=1
    predicted=torch.tensor(predicted)
    predicted=predicted.to(device)
    '''
#    class0Predicted=torch.where(predicted==0,ones,zeros)
#    class1Predicted=torch.where(predicted==1,ones,zeros)
#    class2Predicted=torch.where(predicted==2,ones,zeros)
#    class3Predicted=torch.where(predicted==3,ones,zeros)

    #compute confusion matrixes
    confMclass0=confusionMatrix(class0Truth,class0NeqTruth,prediction[0][0],ones,zeros)
    confMclass1=confusionMatrix(class1Truth,class1NeqTruth,prediction[0][1],ones,zeros)
    #confMclass2=confusionMatrix(class2NeqTruth,class2NeqTruth,prediction[0][1],ones,zeros)
    #confMclass2=confusionMatrix(class3NeqTruth,class3NeqTruth,class3Predicted,ones,zeros)

    #TP,TN,FP,FN
    confMatrix=torch.add(confMclass0,confMclass1)

    try:
        precision=confMatrix[0].item()/(confMatrix[0].item()+confMatrix[2].item())
        recall=confMatrix[0].item()/(confMatrix[0].item()+confMatrix[3].item())
        maxF=2*((precision*recall)/(precision+recall))
    except:
        maxF= 0
    #end=time.time()
    #print("accuracy elapsed time:",end-start)
    return maxF

    #recall=true_positives/(true_positives+false_negatives)
#    maxF=
#    return maxF

def confusionMatrix(classTruth,classNeqTruth,prediction,ones,zeros):

        classTruthPrediction=torch.mul(classTruth,prediction)
        classNeqPrediction=torch.mul(classNeqTruth,prediction)
        classTruthPrediction=torch.mul(classTruth,prediction)
        classNeqPrediction=torch.mul(classNeqTruth,prediction)

        classTPtensor=torch.where(classTruthPrediction>0.5,ones,zeros)
        classTNtensor=torch.where(classNeqPrediction<0.5,ones,zeros)
        classFPtensor=torch.where(classNeqPrediction>0.5,ones,zeros)
        classFNtensor=torch.where(classTruthPrediction<0.5,ones,zeros)

        classTP=classTPtensor.sum()
        classTN=classTNtensor.sum()
        classFP=classFPtensor.sum()
        classFN=classFNtensor.sum()

        classTPtensor=torch.where(classTruthPrediction>0.5,ones,zeros)
        classTNtensor=torch.where(classNeqPrediction<0.5,ones,zeros)
        classFPtensor=torch.where(classNeqPrediction>0.5,ones,zeros)
        classFNtensor=torch.where(classTruthPrediction<0.5,ones,zeros)

        classTP=classTPtensor.sum()
        classTN=classTNtensor.sum()
        classFP=classFPtensor.sum()
        classFN=classFNtensor.sum()

        return torch.stack([classTP,classTN,classFP,classFN])
