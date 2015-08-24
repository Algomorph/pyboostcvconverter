cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(matmul)

find_package(PythonLibs)
find_package(OpenCV)
find_package(Boost)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
  set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  message(WARNING "Check/options for compiler C++11 support required")
  # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message(WARNING "Check/options for compiler C++11 support required")
  # using Visual Studio C++
endif()

message(STATUS ${PYTHON_INCLUDE_DIRS})

add_executable(matmul matmul.cpp)
target_include_directories(matmul PUBLIC 
	${Boost_INCLUDE_DIRS} 
	${OpenCV_INCLUDE_DIRS} 
	${PYTHON_INCLUDE_DIRS} 
)
target_link_libraries(matmul 
	${Boost_LIBRARIES} 
	${OpenCV_LIBRARIES} 
	${PYTHON_LIBRARIES}
)
