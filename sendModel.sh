#!/bin/bash
# -*- coding: utf-8 -*-
#   --------------------------------------------------------------------------------------------
#   |    Author: Michal Zelenak                                                                  |
#   |    BUT Faculty of Information Technology                                                   |
#   |    This is code written for the bachelor thesis                                            |
#   |    Project: Object Detection in the Laser Scans Using Convolutional Neural Networks        |
#   -----------------------------------------------------------------------------------------

#command for sending the model to the school server
scp -r ./Model.tar xzelen24@eva.fit.vutbr.cz:./Backup/
