#!/bin/bash
dirWithStats="stats/"

mainF(){
    statsFile=$1"stats/"
  #  echo $statsFile
#rm -R $statsFile
    #mkdir $dirWithStats

  #  echo $1
    check=$(ls $1| grep "poinCL")
    echo $1
for item in $check; do
        input=$1
        input+=$item
        echo $input

        ./gridMaker.py $input $2
		done
}


rm -R /home/michal/Plocha/bc/stats/train
rm -R /home/michal/Plocha/bc/stats/test
mkdir /home/michal/Plocha/bc/stats/train
mkdir /home/michal/Plocha/bc/stats/test

mainF "pclFiles/test/" "/home/michal/Plocha/bc/stats/test/"
mainF "pclFiles/train/" "/home/michal/Plocha/bc/stats/train/"
