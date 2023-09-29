cmake_minimum_required(VERSION 3.20)

project(stationary_measurement LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Cuda paths and settings
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0")
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Digitizer Spectrum paths
set(SPCM_ROOT_DIR c_header)

# Visa paths
set(VISA_ROOT_DIR "C:/Program Files/IVI Foundation/VISA/Win64")
set(VISA_INCLUDE_DIR ${VISA_ROOT_DIR}/Include)
set(VISA_LIB_DIR ${VISA_ROOT_DIR}/Lib_x64/msc)

# Pybind11 and Python paths
set(CONDA_ROOT_DIR "D:/miniconda3/envs/Measurement")
set(Python_EXECUTABLE ${CONDA_ROOT_DIR}/python.exe)
set(pybind11_DIR ${CONDA_ROOT_DIR}/Library/share/cmake/pybind11)

# Find Pybind11 and Python
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Find cuBLAS, cuFFT, NPPS libraries
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_LIBRARIES})
find_library(CUFFT_LIBRARIES cufft HINTS ${CUDA_LIBRARIES})
find_library(NPPC_LIBRARIES nppc HINTS ${CUDA_LIBRARIES})
find_library(NPPS_LIBRARIES npps HINTS ${CUDA_LIBRARIES})

# Find Spectrum Digitizer Library
find_library(SPCM_LIBRARIES spcm_win64_msvcpp ${SPCM_ROOT_DIR})

# Find VISA Library
find_library(VISA_LIBRARY visa64 ${VISA_LIB_DIR})

# Gather all .cpp, .h, .cu and .cuh files
file(GLOB CPP_SOURCES *.cpp)
file(GLOB CPP_HEADERS *.h)
file(GLOB CUDA_SOURCES *.cu)
file(GLOB CUDA_HEADERS *.cuh)

# Link the libraries to your target (assuming your target is named 'MyCudaProject')
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CPP_HEADERS} ${CUDA_SOURCES} ${CUDA_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS} ${SPCM_ROOT_DIR} ${VISA_INCLUDE_DIR} ${Python_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUBLAS_LIBRARIES} ${CUFFT_LIBRARIES} ${NPPC_LIBRARIES}
                      ${NPPS_LIBRARIES} ${SPCM_LIBRARIES} ${VISA_LIBRARY} ${Python_LIBRARIES})
target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")

# pybind11_add_module(my_python_module bindings.cpp)