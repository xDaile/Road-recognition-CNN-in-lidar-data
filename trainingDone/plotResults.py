#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys
import os


f=open("./1/results.txt","r")
content=f.read()
data=eval(content)

trainLoss       =np.array(data["train"]["Loss"])
trainACCprecise =np.array(data["train"]["Accuracy-precise"] )
trainMaxFprec   =np.array(data["train"]["MaxF-precise"] )
trainAccuracy   =np.array(data["train"]["Accuracy"] )
trainMaxF       =np.array(data["train"]["MaxF"] )
trainVariation  =np.array(data["train"]["VariationOfAccuracy"] )

testLoss       =np.array(data["test"]["Loss"])
testACCprecise =np.array(data["test"]["Accuracy-precise"] )
testMaxFprec   =np.array(data["test"]["MaxF-precise"] )
testAccuracy   =np.array(data["test"]["Accuracy"] )
testMaxF       =np.array(data["test"]["MaxF"] )
testVariation  =np.array(data["test"]["VariationOfAccuracy"] )


fig = plt.figure(figsize=(6, 3.2))
plt.plot(trainMaxF,"green",label="Trénovacia sada")
plt.plot(testMaxF,"blue",label="Testovacia sada")
plt.show()
