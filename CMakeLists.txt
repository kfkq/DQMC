# Set the minimum required version of CMake and define the project.
cmake_minimum_required(VERSION 3.14)
project(DQMC CXX)

# Set the C++ standard to C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set optimized flags for release builds.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# --- Dependencies ---

# Find the Armadillo library. This is a required dependency.
find_package(Armadillo REQUIRED)

# Use FetchContent to download the doctest testing framework automatically.
# This avoids needing to install it system-wide.
include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)


# --- Library Target ---

# Gather all source files from the 'source' directory and its subdirectories.
file(GLOB_RECURSE DQMC_SOURCES "source/*.cpp")

# Create a static library named 'dqmc_lib' from the source files.
add_library(dqmc_lib STATIC ${DQMC_SOURCES})

# Specify the public include directories for the library.
# Any target that links against dqmc_lib will automatically get these include paths.
target_include_directories(dqmc_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link the library to its dependencies.
target_link_libraries(dqmc_lib PUBLIC ${ARMADILLO_LIBRARIES})


# --- Testing ---

# Enable testing for this project.
enable_testing()

# Find all test source files in the 'test' directory.
file(GLOB TEST_FILES "test/*.cpp")

# Loop through each test file and create a corresponding executable.
foreach(TEST_FILE ${TEST_FILES})
  # Generate a clean name for the test executable from its filename.
  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

  # Add the test executable.
  add_executable(test_${TEST_NAME} ${TEST_FILE})

  # Link the test executable against our library and the doctest framework.
  target_link_libraries(test_${TEST_NAME} PRIVATE dqmc_lib doctest::doctest)

  # Register the executable with CTest.
  add_test(NAME ${TEST_NAME}_test COMMAND test_${TEST_NAME})
endforeach()