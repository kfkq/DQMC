cmake_minimum_required(VERSION 3.10)
project(dqmc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C++ compiler to Intel if available
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # Using Intel oneAPI compiler (icx)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ipo -xHost -no-prec-div -fp-model fast=2") 
    
    # -mauto-arch=CORE-AVX2,CORE-AVX512,COMMON-AVX512: Generates code that can run on processors with these instruction sets
    # -fargument-noalias: Assumes that function pointer arguments do not alias with other pointer arguments (optimization)
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-mauto-arch=CORE-AVX2,CORE-AVX512,COMMON-AVX512")
    list(APPEND CMAKE_CXX_FLAGS_RELEASE "-fargument-noalias")
else()
    # Fallback for other compilers (like GCC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
endif()

# Find MPI
find_package(MPI REQUIRED)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Find HDF5
# Check if HDF5_ROOT is set for local installation
if(DEFINED ENV{HDF5_ROOT})
    set(HDF5_ROOT $ENV{HDF5_ROOT})
    message(STATUS "Using local HDF5 installation at: ${HDF5_ROOT}")
else()
    # Try to find HDF5 in standard paths
    find_package(HDF5 REQUIRED COMPONENTS CXX)
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
    ${MPI_CXX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${ARMADILLO_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    main.cpp
    model.cpp
    dqmc.cpp
    observables.cpp
)

# Add executable
add_executable(dqmc ${SOURCES})

# Link libraries
target_link_libraries(dqmc
    ${MPI_CXX_LIBRARIES}
    ${ARMADILLO_LIBRARIES}
    ${MKL_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${HDF5_HL_LIBRARIES}
)

# Set Armadillo to use MKL
target_compile_definitions(dqmc PRIVATE 
    ARMA_USE_MKL_ALLOC
    ARMA_USE_MKL
)