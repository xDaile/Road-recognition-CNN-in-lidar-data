#!/bin/bash
#dorobit tento set
rm -R '/home/michal/Plocha/bc/tensors/train'
rm -R '/home/michal/Plocha/bc/tensors/test'
mkdir '/home/michal/Plocha/bc/tensors/test'
mkdir '/home/michal/Plocha/bc/tensors/train'


python3 createTensors.py


cp -r "/home/michal/Plocha/bc/tensors/train" "/home/michal/Plocha/bc/DatasetForUse"
cp -r "/home/michal/Plocha/bc/tensors/test" "/home/michal/Plocha/bc/DatasetForUse"
cp -r "/home/michal/Plocha/bc/groundTruth/train" "/home/michal/Plocha/bc/DatasetForUse"
cp -r "/home/michal/Plocha/bc/groundTruth/test" "/home/michal/Plocha/bc/DatasetForUse"
