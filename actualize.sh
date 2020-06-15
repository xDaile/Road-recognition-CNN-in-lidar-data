#!/bin/bash




if [[( $# != 1)]]
then
    echo "
        -bin for actualization from bin files
        -pcl for actualization from pcl files
        -stats for actualization from stats
        -tensors for finish actualization only"
    exit 0
fi

if [ $1 == "-bin" ]
then
  echo "Start generating PclFiles"
  bash binToPCLscript.sh
  echo "Start generating Stats"
  bash pclToStats.sh
  echo "Start generating Tensors"
  bash tensorsDirCreate.sh
  echo "Start Finishing"
  bash finishDataset.sh
fi

if [ $1 == "-pcl" ]
then
  echo "Start generating Stats"
  bash pclToStats.sh
  echo "Start generating Tensors"
  bash tensorsDirCreate.sh
  echo "Start Finishing"
  bash finishDataset.sh
fi

if [ $1 == "-stats" ]
then
  echo "Start generating Tensors"
  bash tensorsDirCreate.sh
  echo "Start Finishing"
  bash finishDataset.sh
fi

if [ $1 == "-tensors" ]
then
  echo "Start Finishing"
  bash finishDataset.sh
fi
