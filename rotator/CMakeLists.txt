cmake_minimum_required(VERSION 2.8.11)

project(rotator)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

find_package( PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${SLAM_P_P_DIR}/include ${SLAM_P_P_DIR}/include/cholmod/SuiteSparse)
set(SLAM_P_P_LIBS ${SLAM_P_P_DIR}/lib/libslampp_base.a ${SLAM_P_P_DIR}/lib/libcamd_dlong.a ${SLAM_P_P_DIR}/lib/libcholmod.a
        ${SLAM_P_P_DIR}/lib/libcsparse.a ${SLAM_P_P_DIR}/lib/libcxsparse.a ${SLAM_P_P_DIR}/lib/libeigen.a)

FIND_PACKAGE(Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )


find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR)

FIND_PACKAGE(Boost 1.40 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

set(project_SOURCES createRotation.cpp)

add_executable(${PROJECT_NAME} ${project_SOURCES})

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${ButVELODYNE_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
