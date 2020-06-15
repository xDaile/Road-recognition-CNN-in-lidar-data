#!/bin/bash
rm -R "tensors"
mkdir "./tensors/"
mkdir "./tensors/train"
mkdir "./tensors/test"
python3 statsToTensors.py
