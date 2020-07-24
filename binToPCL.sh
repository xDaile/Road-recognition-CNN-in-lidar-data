#!/bin/bash

#creating pcl files from bin files

mainF(){
  #$1 location of bin files
    pclFiles=$1
    pclFiles+="pclFiles/"
    echo $1
    #recreate dir in current where pclFiles will be saved
    rm -R $pclFiles
    mkdir $pclFiles
    check=$(ls $1| grep "bin")

    #loop over files in dir
    for item in $check; do

        #name of the new file
        newName=${item/bin/poinCL}

        #dir where we are working
        input=$1
        #dir + file for working
        input+=$item

        #input for kitti2pcd
        echo $input

        #dir where we are saving
        output=$pclFiles

        #dir+new name
        output+=$newName
        echo $output

        ./kitti2pcl/bin/kitti2pcd --infile $input --outfile $output
		done
}

#recreate dirs for pclFiles - where new files will be saved
rm -R /home/michal/Plocha/bc/pclFiles/test
rm -R /home/michal/Plocha/bc/pclFiles/train
mkdir /home/michal/Plocha/bc/pclFiles/test
mkdir /home/michal/Plocha/bc/pclFiles/train

mainF "binFiles/testing/"
mainF "binFiles/training/"

#moving created files to another dir, and removing moved ones
mv  -v /home/michal/Plocha/bc/binFiles/training/pclFiles/* /home/michal/Plocha/bc/pclFiles/train
rmdir /home/michal/Plocha/bc/binFiles/training/pclFiles/
mv  -v /home/michal/Plocha/bc/binFiles/testing/pclFiles/* /home/michal/Plocha/bc/pclFiles/test
rmdir /home/michal/Plocha/bc/binFiles/testing/pclFiles/
