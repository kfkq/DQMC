cmake_minimum_required(VERSION 3.10)
project(dqmc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math") 

# Find MPI
find_package(MPI REQUIRED)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find FFTW3
find_path(FFTW_INCLUDE_DIR fftw3.h)
find_library(FFTW_LIBRARY NAMES fftw3)
if(NOT FFTW_LIBRARY)
    message(FATAL_ERROR "FFTW3 library not found")
endif()

# MKL settings
if(DEFINED ENV{MKLROOT})
    set(MKL_ROOT $ENV{MKLROOT})
    set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
    set(MKL_LIB_DIR "${MKL_ROOT}/lib/intel64")
    
    # MKL libraries
    set(MKL_LIBRARIES
        "${MKL_LIB_DIR}/libmkl_intel_lp64.so"
        "${MKL_LIB_DIR}/libmkl_sequential.so"
        "${MKL_LIB_DIR}/libmkl_core.so"
    )
else()
    message(FATAL_ERROR "MKLROOT environment variable is not set")
endif()

# Include directories
include_directories(
    ${MPI_CXX_INCLUDE_PATH}
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
    ${ARMADILLO_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIR}
    ${FFTW_INCLUDE_DIR}
)

# Source files
set(SOURCES
    main.cpp
    model.cpp
    dqmc.cpp
    measurement.cpp
)

# Add executable
add_executable(dqmc ${SOURCES})

# Link libraries
target_link_libraries(dqmc
    ${MPI_CXX_LIBRARIES}
    ${ARMADILLO_LIBRARIES}
    ${MKL_LIBRARIES}
    ${FFTW_LIBRARY}
)

# Set Armadillo to use MKL
target_compile_definitions(dqmc PRIVATE 
    ARMA_USE_MKL_ALLOC
    ARMA_USE_MKL
)
