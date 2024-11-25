cmake_minimum_required(VERSION 3.15...3.27)
project(_eventellipsometer_impl)

if(CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj") # for debug build
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# c++20
set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Enable optimization for core module
# To achieve this, we should run separate compilation steps
# https://nanobind.readthedocs.io/en/latest/api_cmake.html#command:nanobind_add_module
# add_library(lib STATIC src/equations.cpp src/optim.cpp)
# set_target_properties(lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# if(WIN32)
#   set(MY_RELEASE_OPTIONS /Od /O2)
#   # set(MY_RELEASE_OPTIONS /W4 /EHsc /Od /O2 /EIGEN_NO_DEBUG)
# else()
#   set(MY_RELEASE_OPTIONS -Wall -Wextra -Wpedantic -O3)
# endif()

# target_compile_options(lib PRIVATE "$<$<CONFIG:Release>:${MY_RELEASE_OPTIONS}>")



# supress C4819 warning for eigen
if(MSVC)
  add_definitions("/wd4819")
endif()

# Compile extension module with size optimization and add the library
nanobind_add_module(_eventellipsometer_impl NOMINSIZE src/cpp/eventellipsometer.cpp)
# nanobind_add_module(_eventellipsometer_impl NOMINSIZE src/eventellipsometer.cpp src/equations.cpp src/optim.cpp)
# target_link_libraries(_eventellipsometer_impl PRIVATE lib)

# Eigen3
find_package(Eigen3 REQUIRED)
target_include_directories(_eventellipsometer_impl PRIVATE ${EIGEN3_INCLUDE_DIR})
# target_include_directories(lib PRIVATE ${EIGEN3_INCLUDE_DIR})

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(_eventellipsometer_impl PRIVATE OpenMP::OpenMP_CXX)
