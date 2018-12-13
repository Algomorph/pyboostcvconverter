cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project("pbcvt")

#----------------------------CMAKE & GLOBAL PROPERTIES-------------------------#
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

###============= C++11 support====================================
if(${CMAKE_VERSION} VERSION_LESS "3.1")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()
else()
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
#=================================================================
# PYTHON option

set(PYTHON_OPTIONS "2.X" "3.X")
set(PYTHON_DESIRED_VERSION "3.X" CACHE STRING "Choose which python version to use, options are: ${PYTHON_OPTIONS}.")
set_property(CACHE PYTHON_DESIRED_VERSION PROPERTY STRINGS ${PYTHON_OPTIONS})


#=============== Find Packages ====================================
## OpenCV
find_package(OpenCV COMPONENTS core REQUIRED)

## Python
include("DetectPython")

## Boost
if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    # Provisions for typical Boost compiled on Windows
    # Most commonly, Boost libraries are compiled statically on windows (change as necesssary)
    set(Boost_USE_STATIC_LIBS TRUE)
    set(Boost_USE_STATIC_RUNTIME OFF)
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_DEBUG_RUNTIME ON)
	set(Boost_USE_DEBUG_PYTHON OFF)
endif()
if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    set(Python_ADDITIONAL_VERSIONS ${PYTHON2_VERSION_MAJOR}.${PYTHON2_VERSION_MINOR})
    find_package(Boost COMPONENTS python${PYTHON2_VERSION_MAJOR}${PYTHON2_VERSION_MINOR} QUIET)
    if(NOT Boost_FOUND)
       message(STATUS "Trying alternative Boost.Python component name, python-py<version>...")
	     find_package(Boost COMPONENTS python-py${PYTHON2_VERSION_MAJOR}${PYTHON2_VERSION_MINOR} REQUIRED)
     endif()
else ()
    set(Python_ADDITIONAL_VERSIONS ${PYTHON3_VERSION_MAJOR}.${PYTHON3_VERSION_MINOR})
    find_package(Boost COMPONENTS python${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR} QUIET)
    if(NOT Boost_FOUND)
       message(STATUS "Trying alternative Boost.Python component name, python-py<version>...")
	     find_package(Boost COMPONENTS python-py${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR} REQUIRED)
     endif()
endif ()



#========pick python stuff========================================
if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    set(PYTHON_INCLUDE_DIRS ${PYTHON2_INCLUDE_DIR} ${PYTHON2_INCLUDE_DIR2} ${PYTHON2_NUMPY_INCLUDE_DIRS})
    set(PYTHON_NUMPY_INCLUDE_DIRS ${PYTHON2_NUMPY_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${PYTHON2_LIBRARY})
    set(PYTHON_EXECUTABLE ${PYTHON2_EXECUTABLE})
    set(PYTHON_PACKAGES_PATH ${PYTHON2_PACKAGES_PATH})
    set(ARCHIVE_OUTPUT_NAME pbcvt_py2)
else ()
    set(PYTHON_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR} ${PYTHON3_INCLUDE_DIR2} ${PYTHON3_NUMPY_INCLUDE_DIRS})
    set(PYTHON_NUMPY_INCLUDE_DIRS ${PYTHON3_NUMPY_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${PYTHON3_LIBRARY})
    set(PYTHON_EXECUTABLE ${PYTHON3_EXECUTABLE})
    set(PYTHON_PACKAGES_PATH ${PYTHON3_PACKAGES_PATH})
    set(ARCHIVE_OUTPUT_NAME pbcvt_py3)
endif ()

find_package_handle_standard_args(Python DEFAULT_MSG PYTHON_INCLUDE_DIRS PYTHON_NUMPY_INCLUDE_DIRS PYTHON_LIBRARIES PYTHON_EXECUTABLE PYTHON_PACKAGES_PATH)
if(NOT Python_FOUND)
    message(SEND_ERROR "Not all requred components of Numpy/Python found.")
endif()

file(GLOB project_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${project_sources} ${CMAKE_CURRENT_SOURCE_DIR}/include/pyboostcvconverter/pyboostcvconverter.hpp)
target_include_directories(${PROJECT_NAME} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        )

target_link_libraries(${PROJECT_NAME}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${PYTHON_LIBRARIES}
        )

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    # Provisions for typical Boost compiled on Windows
    # Unless some extra compile options are used on Windows, the libraries won't have prefixes (change as necesssary)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DBOOST_ALL_NO_LIB -DBOOST_SYSTEM_NO_DEPRECATED)
endif()

#---------------------------   INSTALLATION    -----------------------------------------------------
#-get proper extension for python binary shared object on this platform

set(__python_ext_suffix_var "EXT_SUFFIX")
if({PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    set(__python_ext_suffix_var "SO")
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('${__python_ext_suffix_var}'))"
        RESULT_VARIABLE PYTHON_${PROJECT_NAME}_PY_PROCESS
        OUTPUT_VARIABLE ${PROJECT_NAME}_PY_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ${PYTHON_${PROJECT_NAME}_PY_PROCESS} EQUAL 0)
    set(${PROJECT_NAME}_PY_SUFFIX ".so")
endif()


set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_NAME ${ARCHIVE_OUTPUT_NAME}  # prevent name conflict for python2/3 outputs
        PREFIX ""
        OUTPUT_NAME pbcvt
        SUFFIX ${${PROJECT_NAME}_PY_SUFFIX})



if (MSVC AND NOT PYTHON_DEBUG_LIBRARIES)
    set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else ()
    set(PYTHON_INSTALL_CONFIGURATIONS "")
endif ()

if (WIN32)
    set(PYTHON_INSTALL_ARCHIVE "")
else ()
    set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python)
endif ()

install(TARGETS ${PROJECT_NAME}
        ${PYTHON_INSTALL_CONFIGURATIONS}
        RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        ${PYTHON_INSTALL_ARCHIVE}
        )
