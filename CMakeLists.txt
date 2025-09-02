cmake_minimum_required(VERSION 3.10)
project(dqmc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math") 

# Find MPI
find_package(MPI REQUIRED)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)

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
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${ARMADILLO_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# Source files for main executable
set(MAIN_SOURCES
    source/main.cpp
    source/model.cpp
    source/dqmc.cpp
    source/update.cpp
    source/stablelinalg.cpp
)

# Add main executable
add_executable(dqmc ${MAIN_SOURCES})

# Link libraries for main executable
target_link_libraries(dqmc
    ${MPI_CXX_LIBRARIES}
    ${ARMADILLO_LIBRARIES}
    ${MKL_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
)

# Set Armadillo to use MKL for main executable
target_compile_definitions(dqmc PRIVATE 
    ARMA_USE_MKL_ALLOC
    ARMA_USE_MKL
)
