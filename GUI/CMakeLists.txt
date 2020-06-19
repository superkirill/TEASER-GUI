cmake_minimum_required(VERSION 3.10)
project(teaser)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(teaserpp REQUIRED)

# Change this line to include your own executable file
add_executable(teaser teaser.cc)

# Link to teaserpp & Eigen3
target_link_libraries(teaser Eigen3::Eigen teaserpp::teaser_registration teaserpp::teaser_io)