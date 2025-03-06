cmake_minimum_required(VERSION 3.14)

add_executable(
  hubbard1d
  main.cpp
)

target_link_libraries(
  hubbard1d
  PRIVATE
  qmutils
)
