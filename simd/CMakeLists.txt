cmake_minimum_required(VERSION 2.8)

# set the project name
project(Autocorrelate CXX)

add_compile_options(-std=c++11)

# add the executable
add_executable(Autocorrelate main.cc test_rig.cc)