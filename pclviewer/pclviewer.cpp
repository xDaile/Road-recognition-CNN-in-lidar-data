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

	//check if file exists
	FILE *pFile;
	pFile=fopen(argv[1],"r");
	if(pFile==NULL){
		printf("cannot open the file \n");
		return 1;
	}
	fclose(pFile);

	//file was opened
	std::cerr << "Processing KITTI file: " << argv[1] << std::endl << std::flush;

//name of the file
	string infile=argv[1];

//create the cloud
	but_velodyne::VelodynePointCloud cloud;

//fill the cloud
	but_velodyne::VelodynePointCloud::VelodynePointCloud::fromFile(infile, cloud, true);

//create Visualizer
	but_velodyne::Visualizer3D vis;
	//vis.getViewer()->setBackgroundColor(0,0,0);
	//show the cloud
	vis.addPointCloud(cloud).show();

	return 0;
}
