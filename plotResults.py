#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
        -------------------------------------------------------------------------------------------
        |    Author: Michal Zelenak                                                               |
        |    BUT Faculty of Information Technology                                                |
        |    This is code written for the bachelor thesis                                         |
        |    Project: Object Detection in the Laser Scans Using Convolutional Neural Networks     |
        -------------------------------------------------------------------------------------------
"""
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
from pathlib import *
import re
import toolsForNetwork
import math

f= open("./results.txt","r")

data=eval(f.read())

fig = plt.figure(figsize=(6, 3.2))
data["train"]["MaxF"]=data["train"]["MaxF"][0:180]
data["train"]["Accuracy"]=data["train"]["Accuracy"][0:180]
data["test"]["MaxF"]=data["test"]["MaxF"][0:180]
data["test"]["Accuracy"]=data["test"]["Accuracy"][0:180]
data["train"]["MaxF-precise"]=data["train"]["MaxF-precise"][0:180]
data["train"]["Accuracy-precise"]=data["train"]["Accuracy-precise"][0:180]
data["test"]["MaxF-precise"]=data["test"]["MaxF-precise"][0:180]
data["test"]["Accuracy-precise"]=data["test"]["Accuracy-precise"][0:180]
print(len(data["train"]["MaxF"]))
epochs=[9, 18, 27, 36,45,54,63,72,81,90,99,108,117,126,135,144,153,162,171,180]

#plt.xlim(0,200)
#plt.stem(epochs,val)
for val in epochs:
    plt.plot([val,val],[0,1],"gray")
plt.xlim(-5,185)
plt.ylim(0.2,1)
plt.plot(data["train"]["Accuracy"],"green")
plt.plot(data["test"]["Accuracy"],"blue")
plt.show()
