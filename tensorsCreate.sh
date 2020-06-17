#!/bin/bash

#recreating dirs for tensors probably would be eough rm "./tensors/train/*" and rm"./tensors/test/*"
rm -R "tensors"
mkdir "./tensors/"
mkdir "./tensors/train"
mkdir "./tensors/test"

#running script for creating tensors from tstats
python3 createTensors.py
