/*
        -------------------------------------------------------------------------------------------
        |    Author: Michal Zelenak                                                               |
        |    BUT Faculty of Information Technology                                                |
        |    This is code written for the bachelor thesis                                         |
        |    Project: Object Detection in the Laser Scans Using Convolutional Neural Networks     |
        -------------------------------------------------------------------------------------------
*/



#include <stdio.h>
#include <fstream>
#include <string>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <cmath>
#include <pcl/point_types.h>
#include <but_velodyne/VelodynePointCloud.h>
#include <but_velodyne/Visualizer3D.h>
#include <Eigen/StdVector>
using namespace std;
using namespace pcl;


int main(int argc, char *argv[])
{
      //fill the cloud
      pcl::PointCloud< pcl::_PointXYZRGB  > inCloud;

      cout<<"input created"<<endl;
      pcl::PCDReader reader;

      PointCloud<PointXYZRGB>::Ptr cloudPTR(new PointCloud<PointXYZRGB>);
      reader.read(argv[1],*cloudPTR);
      //cloudPTR=&inCloud;
      but_velodyne::Visualizer3D vis;
      vis.setPointSize(2);
      vis.getViewer()->setCameraPosition(-6,0,4,0,0,1);

      //show the cloud
      vis.addColorPointCloud(cloudPTR).show();
}
