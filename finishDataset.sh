#!/bin/bash

#removing old tensors
rm -R '/home/michal/Plocha/bc/tensors/train'
rm -R '/home/michal/Plocha/bc/tensors/test'
mkdir '/home/michal/Plocha/bc/tensors/test'
mkdir '/home/michal/Plocha/bc/tensors/train'

#creating new tensors
python3 createTensors.py

#creating new GT Tensors
python3 groundTruthTensorMaker.py

#copy the new files
cp -r "/home/michal/Plocha/bc/tensors/train" "/home/michal/Plocha/bc/DatasetForUse"
cp -r "/home/michal/Plocha/bc/tensors/test" "/home/michal/Plocha/bc/DatasetForUse"
cp -r "/home/michal/Plocha/bc/groundTruth/train" "/home/michal/Plocha/bc/DatasetForUse"
cp -r "/home/michal/Plocha/bc/groundTruth/test" "/home/michal/Plocha/bc/DatasetForUse"
