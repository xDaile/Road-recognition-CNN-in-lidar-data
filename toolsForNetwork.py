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


#cuda device switch to nvidia
def get_device():
    global cuda0
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        cuda0=torch.device('cuda')

        print("Device changed to: "+ torch.cuda.get_device_name(0))
        return cuda0
    else:
        print("device rtx 2080ti was not found, rewrite baseCNN or parameters")
        exit(1)
