cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)

project(Cluster++ CXX)

set(Eigen_INCLUDE_DIR /home/nicolas/opt/Eigen/include/eigen3)

file(GLOB Cluster++_SRC
    "src/*.cpp"
    "src/*.hpp"
    "src/algorithms/*.hpp"
    "src/algorithms/*.cpp"
    "src/metrics/*.cpp"
    "src/metrics/*.hpp"
    "src/utils/*.cpp"
    "src/utils/*.hpp"
    "tests/*.hpp"
    "tests/*.cpp"
)

include_directories(${Eigen_INCLUDE_DIR})

add_executable(Cluster++ ${Cluster++_SRC})

SET(CMAKE_CXX_FLAGS "-O3 -fopenmp -std=c++11")