cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)

set(PROJECT_NAME ClusterXX-Test)

project(${PROJECT_NAME} CXX)

set(EIGEN_INCLUDE_DIR /home/nicolas/opt/Eigen/include/eigen3)

set(CLUSTERXX_ROOT_DIR /home/nicolas/dev/eclipse-workspace/ClusterXX)

set(CLUSTERXX_INCLUDE_DIR ${CLUSTERXX_ROOT_DIR}/includes)

set(CMAKE_CXX_FLAGS "-O3 -std=c++14 -fopenmp")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_BINARY_DIR}/lib)

file(GLOB CLUSTERXX_SRC_FILES
    "${CLUSTERXX_ROOT_DIR}/src/*.cpp"    
    "${CLUSTERXX_ROOT_DIR}/src/algorithms/*.cpp"
    "${CLUSTERXX_ROOT_DIR}/src/metrics/*.cpp"
    "${CLUSTERXX_ROOT_DIR}/src/utils/*.cpp"
)

file(GLOB CLUSTERXX_INCLUDE_FILES
    "${CLUSTERXX_ROOT_DIR}/includes/*.hpp"    
    "${CLUSTERXX_ROOT_DIR}/includes/algorithms/*.hpp"
    "${CLUSTERXX_ROOT_DIR}/includes/metrics/*.hpp"
    "${CLUSTERXX_ROOT_DIR}/includes/utils/*.hpp"
)

file(GLOB PROJECT_SRC
    "tests/*.hpp"
    "tests/*.cpp"
)

include_directories(${EIGEN_INCLUDE_DIR} ${CLUSTERXX_INCLUDE_DIR})

add_library(ClusterXX STATIC ${CLUSTERXX_SRC_FILES} ${CLUSTERXX_INCLUDE_FILES})

add_executable(${PROJECT_NAME} ${PROJECT_SRC})
target_link_libraries(${PROJECT_NAME} ClusterXX)