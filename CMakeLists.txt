cmake_minimum_required(VERSION 3.24)
project(NeuralNetwork2)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetwork2 main.cpp)
set(CMAKE_EXE_LINKER_FLAGS "-static")