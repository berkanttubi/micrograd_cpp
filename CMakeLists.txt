cmake_minimum_required(VERSION 3.22)
project(microgradcpp_beko)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

file(GLOB_RECURSE SOURCE src/*.cpp)

add_executable(microgradcpp_beko ${SOURCE})

target_include_directories(microgradcpp_beko PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include

        )