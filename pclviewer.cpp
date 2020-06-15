#include <but_velodyne/VelodynePointCloud.h>
#include <but_velodyne/Visualizer3D.h>

#include <stdio.h>

//using namespace ;
int main(int argc, char *argv[])
{
	//if there is not one argument -h or name of the file with lidar points
	if(argc!=2){
		printf("Print help with -h\n");
		return 1;
	}

	//if there is argument for help,print help, end the program
	if(argv[1][1]=='h'){
		printf("Visualizer for lidar points cloud\n");
		printf("Usage: ./visualizer <name of the file with data>\n");
		return 0;
	}

	//try to open file

	FILE *pFile;
	pFile=fopen(argv[1],"r");
	if(pFile==NULL){
		printf("cannot open the file \n");
		return 1;
	}
	fclose(pFile);

//printf("%s\n",*pFile);
	//file was opened
//h	but_velodyne::VelodynePointCloud out_cloud;
std::cerr << "Processing KITTI file: " << argv[1] << std::endl << std::flush;

string infile=argv[1];
but_velodyne::VelodynePointCloud cloud;
but_velodyne::VelodynePointCloud::VelodynePointCloud::fromFile(infile, cloud, true);
/*
		 pcl::io::loadPCDFile(argv[1], cloud);
		 cloud.setImageLikeAxisFromKitti();
		 cloud.estimateModel();


*/

	//create Visualizer
but_velodyne::Visualizer3D vis;

	//plot the image
vis.addPointCloud(cloud).show();

	//printf("done\n");
  return 0;
}
