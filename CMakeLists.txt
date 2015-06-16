cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)

project(Cluster++ CXX)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/lib)
set(EIGEN_INCLUDE_DIR /home/nicolas/opt/Eigen/include/eigen3)
set(LOCAL_INCLUDE_DIR ./includes)
SET(CMAKE_CXX_FLAGS "-O3 -fopenmp -std=c++14")

file(GLOB LIBRARY_SRC
    "src/*.cpp"    
    "src/algorithms/*.cpp"
    "src/metrics/*.cpp"
    "src/utils/*.cpp"
)

file(GLOB LIBRARY_INCLUDES
    "includes/*.hpp"    
    "includes/algorithms/*.hpp"
    "includes/metrics/*.hpp"
    "includes/utils/*.hpp"
)
 
 file(GLOB TEST_SRC
 	"tests/*.hpp"
    "tests/*.cpp"
 ) 	

include_directories(${EIGEN_INCLUDE_DIR} ${LOCAL_INCLUDE_DIR}) 
include(lib/CMakeLists.txt)

