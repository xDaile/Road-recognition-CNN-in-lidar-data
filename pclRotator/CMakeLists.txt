cmake_minimum_required(VERSION 2.8.11)

project(pclRotator)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)



find_package( PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



set(project_SOURCES pclRotator.cpp)

add_executable(${PROJECT_NAME} ${project_SOURCES})

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${ButVELODYNE_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
