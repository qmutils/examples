cmake_minimum_required(VERSION 3.14)

find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)

add_executable(
  kagome_quench
  main.cpp
)

target_link_libraries(
  kagome_quench
  PRIVATE
  qmutils
  OpenMP::OpenMP_CXX
  armadillo
)
