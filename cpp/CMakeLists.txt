cmake_minimum_required(VERSION 3.5.1)
project(lab_01)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lab_01_eigen)
add_subdirectory(lab_01_transformations)
