cmake_minimum_required(VERSION 2.8.11)

project(pclResultView)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package( PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

find_package(but_velodyne REQUIRED)
include_directories(${ButVELODYNE_INCLUDE_DIRS})
link_directories(${ButVELODYNE_LIBRARY_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)

FIND_PACKAGE(Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

set(project_SOURCES pclResultView.cpp) # showResult.cpp Repair.cpp

add_executable(${PROJECT_NAME} ${project_SOURCES})

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${ButVELODYNE_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
