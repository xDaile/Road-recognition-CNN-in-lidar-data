#!/bin/bash

#creating stats from pclFiles
mainF(){

    check=$(ls $1)
    #echo $check
    #loop throught pclFiles
    for item in $check; do
        #input = name of the dir -$1 + file that is actually readen
        input=$1
        input+=$item
        echo $input

        #making stats
        ./gtFlip.py $input
		done
}

#recreating the dirs for the stats
#rm -R /home/michal/Plocha/bc/stats/train
#rm -R /home/michal/Plocha/bc/stats/test
#mkdir /home/michal/Plocha/bc/stats/train
#mkdir /home/michal/Plocha/bc/stats/test


mainF "./groundTruthTensors/test/"
mainF "./groundTruthTensors/train/"
