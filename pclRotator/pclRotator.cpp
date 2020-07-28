#include <stdio.h>
#include <fstream>
#include <string>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <cmath>
#include <pcl/point_types.h>


#define ROTATIONDIRECTIONPLUS 10000
#define ROTATIONDIRECTIONMINUS 20000
#define ANGLECONSTANT 100
#define NAMEOFFILESTART 30


const int positionOfNumberInFileName=30;
const int lenghtOfEndingOfFile=7;
using namespace std;
using namespace pcl;

//WARNING!!!! Format pointDEM is used only for its compatibility with our needs, we need format which can store x,y,z,i,Class,Flag,

float getRadFromDegree(double degree){
  return float((float(degree)*M_PI)/180);
  }


//Scheme for new number of file: ABCDEF
// A - file is flipped by A=1, A=0 file is not flipped
// B - file is rotated by positive(1) or negative(2) angles
// CD - angle of rotation from set of angles -30,-27,-24,-21,-18,-15,-12,-9,-6,-3,3,6,9,12,15,18,21,24,27,30
// EF - original numbers from dataset
string getNewName(char oldName[],int angle){

  std::string my_str(oldName);

  //because of the difference of the names of pclFiles umm_,um_,uu_
  int shift=0;
  string typeOfLoadedFile;
  if(my_str.find("umm")!=string::npos){
    shift=1;
    typeOfLoadedFile="umm";
    }
  else if(my_str.find("uu")!=string::npos){
    typeOfLoadedFile="uu";
    }
  else
    typeOfLoadedFile="um";

  //extract directory
  string dir=my_str.substr(0,NAMEOFFILESTART); //HAVE DIR

  //get the type of the name of pclFile, probably do not need that i have shift
  //  string typeOfPcl=my_str.substr(NAMEOFFILESTART,3);//umm,um_, uu_ are possibilities

  //number of file +3 because of umm_/um_/uu_
  string numberOfFileStr=my_str.substr(NAMEOFFILESTART+3+shift,7);
  //get only numbers
  numberOfFileStr=numberOfFileStr.substr(1,7-shift);


  //we will change that number, depending on rotations
  int numberOfFile=std::stoi(numberOfFileStr);
  //count new number, see the doc for this function
  if(angle<0)
    {
      numberOfFile+=ROTATIONDIRECTIONMINUS+angle*(-ANGLECONSTANT);
    }
  else{
    numberOfFile+=ROTATIONDIRECTIONPLUS+angle*ANGLECONSTANT;
    }

  string newNumberOfFile=to_string(numberOfFile);

  int lenOfNumber=newNumberOfFile.length();

  while(lenOfNumber<6){
    newNumberOfFile="0"+newNumberOfFile;
    lenOfNumber++;
    }

  dir=dir.replace(10,31,"/rotatedPCL/");
  string newName=dir+typeOfLoadedFile+"_"+newNumberOfFile+".pcd";
  return newName;
  }

//using namespace pcl;
int main(int argc, char *argv[])
{

  int angles[]={-30,-27,-24,-21,-18,-15,-12,-9,-6,-3,0,3,6,9,12,15,18,21,24,27,30};
  //double angles[]={-4.5,-3,-1.5,0,1.5,3,4.5}; model get 73.25 precision with these angles,
  //int angles[]={0,3};
  int numOfAngles=sizeof(angles)/sizeof(int);
  int currentAnglesIndex=0;

//save zero rotation into folder

  while(currentAnglesIndex<numOfAngles){
      float angle=getRadFromDegree(angles[currentAnglesIndex]);

      //WARNING!!!! Format pointDEM is used only for its compatibility with our needs, we need format which can store x,y,z,i,Class,Flag,
      //PointDEM format is  (float _x, float _y, float _z, float _intensity, float _intensity_variance, float _height_variance),
      // we are using _intensity_variance as a storage for the ground truth, and _height_variance as a storage for flag (flag that point is faked)
      pcl::PointCloud< pcl::_PointDEM  > inCloud;
      pcl::PointCloud< pcl::_PointDEM  > outCloud;
      pcl::PCDReader reader;
      reader.read(argv[1],inCloud);
      if(angle==0)
      {
        string newName=getNewName(argv[1],angles[currentAnglesIndex]);
        pcl::PCDWriter writer;
        writer.write (newName, inCloud, false);
        currentAnglesIndex++;
        continue;
      }

      //transformation matrix
      Eigen::Affine3f t;

      //get transformation arround angle
      pcl::getTransformation(0,0,0, 0,0,angle,t);


      //transformation matrix print
    //  printf("|%f  %f  %f\n|%f  %f  %f\n|%f  %f  %f\n",t(0,0),t(0,1),t(0,2),t(1,0),t(1,1),t(1,2),t(2,0),t(2,1),t(2,2));

      //apply transformation to cloud
      pcl::transformPointCloud(inCloud, outCloud, t);

      //saveCloud
      string newName=getNewName(argv[1],angles[currentAnglesIndex]);
      pcl::PCDWriter writer;
    //  cout<<newName<<endl;
      writer.write (newName, outCloud, binary=false);
      currentAnglesIndex++;
      //cout<<"okay"<<endl;
    }
}
