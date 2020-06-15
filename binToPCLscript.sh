#!/bin/bash

mainF(){
    pclFiles=$1
    pclFiles+="pclFiles/"
    echo $1
    rm -R $pclFiles
    mkdir $pclFiles
    check=$(ls $1| grep "bin")

    for item in $check; do
        newName=${item/bin/poinCL}


        input=$1
        input+=$item
        echo $input

        output=$pclFiles
        output+=$newName
        echo $output
        ./kitti-pcl-master/bin/kitti2pcd --infile $input --outfile $output
		done
}


rm -R /home/michal/Plocha/bc/pclFiles/test
rm -R /home/michal/Plocha/bc/pclFiles/train
mkdir /home/michal/Plocha/bc/pclFiles/test
mkdir /home/michal/Plocha/bc/pclFiles/train

mainF "binFiles/testing/"
mainF "binFiles/training/"

mv  -v /home/michal/Plocha/bc/binFiles/training/pclFiles/* /home/michal/Plocha/bc/pclFiles/train
rmdir /home/michal/Plocha/bc/binFiles/training/pclFiles/
mv  -v /home/michal/Plocha/bc/binFiles/testing/pclFiles/* /home/michal/Plocha/bc/pclFiles/test
rmdir /home/michal/Plocha/bc/binFiles/testing/pclFiles/
#cd /home/michal/Plocha/bc/pclFiles
