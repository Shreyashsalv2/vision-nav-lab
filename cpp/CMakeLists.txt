cmake_minimum_required(VERSION 3.16)
project(vnav_frames LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VNAV_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(VNAV_BUILD_PYTHON "Build the pybind11 module vnav._frames" OFF)

add_library(vnav_frames STATIC src/frames.cpp)
target_include_directories(vnav_frames PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(vnav_frames PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(vnav_frames PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(frames_cli src/frames_cli.cpp)
target_link_libraries(frames_cli PRIVATE vnav_frames)

if(VNAV_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(frames_test tests/test_frames.cpp)
  target_link_libraries(frames_test PRIVATE vnav_frames GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(frames_test)
endif()

if(VNAV_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_frames src/bindings.cpp)
  target_link_libraries(_frames PRIVATE vnav_frames)
endif()
