cmake_minimum_required(VERSION 3.21)
project(SubPixEdgeContour)

set(CMAKE_CXX_STANDARD 20)

find_package(fmt REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV 4.5.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(TBB REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(main main.cpp)
target_link_libraries(main
        fmt::fmt
        Eigen3::Eigen
        ${OpenCV_LIBS}
        tbb
        Boost::boost
        ${Boost_LIBRARIES}
        )
