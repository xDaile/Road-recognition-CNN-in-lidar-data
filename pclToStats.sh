#!/bin/bash

#creating stats from pclFiles
mainF(){

    check=$(ls $1| grep "poinCL")

    #loop throught pclFiles
    for item in $check; do
        #input = name of the dir -$1 + file that is actually readen
        input=$1
        input+=$item
        echo $input

        #making stats
        ./gridMaker.py $input $2
		done
}

#recreating the dirs for the stats
rm -R /home/michal/Plocha/bc/stats/train
rm -R /home/michal/Plocha/bc/stats/test
mkdir /home/michal/Plocha/bc/stats/train
mkdir /home/michal/Plocha/bc/stats/test


mainF "pclFiles/test/" "/home/michal/Plocha/bc/stats/test/"
mainF "pclFiles/train/" "/home/michal/Plocha/bc/stats/train/"
