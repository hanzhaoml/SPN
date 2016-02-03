cmake_minimum_required(VERSION 2.8)
project(SPN_EM)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

## Executables for batch learning algorithms
set(SOURCE_FILES_BATCH batch_learning.cpp src/SPNNode.cpp src/SPNNode.h src/utils.h src/utils.cpp
    src/SPNetwork.cpp src/SPNetwork.h src/fmath.hpp src/BatchParamLearning.cpp src/BatchParamLearning.h)

## Executables for online learning algorithms
set(SOURCE_FILES_ONLINE online_learning.cpp src/SPNNode.cpp src/SPNNode.h src/utils.h src/utils.cpp
    src/SPNetwork.cpp src/SPNetwork.h src/fmath.hpp src/OnlineParamLearning.cpp src/OnlineParamLearning.h)

## Executable for streaming learning algorithms
set(SOURCE_FILES_STREAM stream_learning.cpp src/SPNNode.cpp src/SPNNode.h src/utils.h src/utils.cpp
    src/SPNetwork.cpp src/SPNetwork.h src/fmath.hpp src/StreamParamLearning.cpp src/StreamParamLearning.h)


# BOOST support
find_package(Boost 1.55.0 COMPONENTS system filesystem thread program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "/Users/hanzhao/ClionProjects/SPN-EM/")

# Online learning
add_executable(online_learning ${SOURCE_FILES_ONLINE})
# Batch learning
add_executable(batch_learning ${SOURCE_FILES_BATCH})
# Stream learning
add_executable(stream_learning ${SOURCE_FILES_STREAM})


# Link libs
target_link_libraries(online_learning ${Boost_LIBRARIES})
target_link_libraries(batch_learning ${Boost_LIBRARIES})
target_link_libraries(stream_learning ${Boost_LIBRARIES})
