cmake_minimum_required(VERSION 3.14)

find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)

add_executable(
  hubbard1d
  main.cpp
)

target_link_libraries(
  hubbard1d
  PRIVATE
  qmutils
  OpenMP::OpenMP_CXX
  armadillo
)
