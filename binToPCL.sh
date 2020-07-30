#!/bin/bash
# -*- coding: utf-8 -*-
#   --------------------------------------------------------------------------------------------
#   |    Author: Michal Zelenak                                                                  |
#   |    BUT Faculty of Information Technology                                                   |
#   |    This is code written for the bachelor thesis                                            |
#   |    Project: Object Detection in the Laser Scans Using Convolutional Neural Networks        |
#   -----------------------------------------------------------------------------------------


#creating pcl files from bin files

mainF(){
  #$1 location of bin files
    pclFiles+="pclFiles/"

    #recreate dir in current where pclFiles will be saved

    check=$(ls $1| grep "bin")

    #loop over files in dir
    for item in $check; do

        #name of the new file
        newName=${item/bin/pcd}

        #dir where we are working
        input=$1
        #dir + file for working
        input+=$item

        #input for kitti2pcd
#        echo $input

        #dir where we are saving
        output=$pclFiles

        #dir+new name
        output+=$newName
    #    echo $output

        ./kitti2pcl/bin/kitti2pcd --infile $input --outfile $output
		done
}


mainF "binFiles/"
