cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)

set(PROJECT_NAME ClusterXX-Test)

project(${PROJECT_NAME} CXX)

set(EIGEN_INCLUDE_DIR $ENV{HOME}/opt/Eigen/include/eigen3)

set(CLUSTERXX_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(CLUSTERXX_INCLUDE_DIR ${CLUSTERXX_ROOT_DIR}/includes)

set(LODEPNG_ROOT_DIR ${CLUSTERXX_ROOT_DIR}/lib/lodepng)
set(LODEPNG_INCLUDE_DIR ${LODEPNG_ROOT_DIR})

set(CMAKE_CXX_FLAGS "-Wall -O3 -std=c++14 -fopenmp")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/lib)

file(GLOB CLUSTERXX_SRC_FILES
    "${CLUSTERXX_ROOT_DIR}/src/*.cpp"    
    "${CLUSTERXX_ROOT_DIR}/src/clustering/*.cpp"
    "${CLUSTERXX_ROOT_DIR}/src/metrics/*.cpp"
    "${CLUSTERXX_ROOT_DIR}/src/utils/*.cpp"
    "${CLUSTERXX_ROOT_DIR}/includes/*.hpp"    
    "${CLUSTERXX_ROOT_DIR}/includes/clustering/*.hpp"
    "${CLUSTERXX_ROOT_DIR}/includes/metrics/*.hpp"
    "${CLUSTERXX_ROOT_DIR}/includes/utils/*.hpp"
)

file(GLOB LODEPNG_SRC_FILES
    "${LODEPNG_ROOT_DIR}/*.cpp"    
    "${LODEPNG_ROOT_DIR}/*.hpp" 
)

file(GLOB PROJECT_SRC
    "tests/*.hpp"
    "tests/*.cpp"
)

include_directories(${EIGEN_INCLUDE_DIR} ${CLUSTERXX_INCLUDE_DIR} ${LODEPNG_INCLUDE_DIR})

add_library(LodePNG STATIC ${LODEPNG_SRC_FILES})
add_library(ClusterXX STATIC ${CLUSTERXX_SRC_FILES})
target_link_libraries(ClusterXX LodePNG)

add_executable(${PROJECT_NAME} ${PROJECT_SRC})
target_link_libraries(${PROJECT_NAME} ClusterXX)
