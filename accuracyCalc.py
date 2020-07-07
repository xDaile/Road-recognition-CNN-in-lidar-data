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

def accuracy(truth, prediction,cuda0):
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
    zeros=zeros.to(device=cuda0)

    ones=torch.ones(1,400,200)
    ones=ones.to(device=cuda0)

    truth=truth.float()

#Split truth to 3 tensors for each class
    class0Truth=torch.where(truth==0,ones,zeros)
    class1Truth=torch.where(truth==1,ones,zeros)
    class2Truth=torch.where(truth==2,ones,zeros)
    #class3Truth=torch.where(truth==3,ones,zeros)


# create the negation of the tensors created before this
    class0NeqTruth=torch.where(truth!=0,ones,zeros)
    class1NeqTruth=torch.where(truth!=1,ones,zeros)
    class2NeqTruth=torch.where(truth!=2,ones,zeros)
#    class3NeqTruth=torch.where(truth!=3,ones,zeros)

    #points where class is 3 are zero, otherwise 1, we will multiply by that tensor rest of the tensors

#    class0NeqTruth=torch.mul(class0NeqTruth,class2PointsZeros)#
#    class1NeqTruth=torch.mul(class1NeqTruth,class2PointsZeros)

    class2PointsZeros=torch.where(truth==2,zeros,ones)
    #compute confusion matrixes
    confMclass0=confusionMatrix(class0Truth,class0NeqTruth,prediction[0][0],ones,zeros,cuda0,class2PointsZeros)
    confMclass1=confusionMatrix(class1Truth,class1NeqTruth,prediction[0][1],ones,zeros,cuda0,class2PointsZeros)
    #confMclass2=confusionMatrix(class2NeqTruth,class2NeqTruth,prediction[0][1],ones,zeros)
    #confMclass2=confusionMatrix(class3NeqTruth,class3NeqTruth,class3Predicted,ones,zeros)

    #TP,TN,FP,FN
    confMatrix=torch.add(confMclass0,confMclass1)
    try:
        precision=confMatrix[0].item()/(confMatrix[0].item()+confMatrix[2].item())
        recall=confMatrix[0].item()/(confMatrix[0].item()+confMatrix[3].item())
        maxF=2*((precision*recall)/(precision+recall))
        accuracy=(confMatrix[0].item()+confMatrix[1].item())/(confMatrix[0].item()+confMatrix[1].item()+confMatrix[2].item()+confMatrix[3].item())

    except:
        maxF= 0
        accuracy=0
    #end=time.time()
    #print("accuracy elapsed time:",end-start)
    return maxF,accuracy

    #recall=true_positives/(true_positives+false_negatives)
#    maxF=
#    return maxF

def confusionMatrix(classTruth,classNeqTruth,prediction,ones,zeros,cuda0,class2PointsZeros):
        #print("Neq",classNeqTruth.sum().item())
        classNeqTruth=torch.mul(classNeqTruth,class2PointsZeros)
        prediction=torch.stack([prediction]).to(device=cuda0)

        #print("Truth a neqTruth",classTruth.sum().item(),classNeqTruth.sum().item())
        classTruthPrediction=torch.mul(classTruth,prediction)

        classNeqPrediction=torch.mul(classNeqTruth,prediction)#EDITED -removed 2 class points
    #    print("neqTruth:",classNeqTruth[0],"\tprediction:",prediction[0],"\tclassNeqPrediction:",classNeqPrediction[0])
        classTPtensor=torch.where(classTruthPrediction>0.5,ones,zeros)
        classTNtensor=torch.where(classNeqPrediction<0.5 and classNeqPrediction!=0,ones,zeros)
    #    print("classTNtensor",classTNtensor[0])

        classFPtensor=torch.where(classNeqPrediction>0.5,ones,zeros)

        FNtensor=torch.where(classTruth==0,ones,zeros)
        FNtensor=torch.mul(FNtensor,prediction)
        classFNtensor=torch.where(FNtensor<0.5,ones,zeros)
        #print("prediction",prediction[0])
        #print("TN",classTNtensor[0])

        classTP=classTPtensor.sum()
        classTN=classTNtensor.sum()
        classFP=classFPtensor.sum()
        classFN=classFNtensor.sum()
        print("TP: ",classTP.item(),"\tTN:",classTN.item(),"\tFP:",classFP.item(),"\tFN:",classFN.item())

        #print("ONE OUTPUT",classTP,classTN,classFP,classFN,"END")

        return torch.stack([classTP,classTN,classFP,classFN])
