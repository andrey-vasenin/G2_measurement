cmake_minimum_required(VERSION 3.20)

project(filtered_psd2 LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Cuda Toolkit
find_package(CUDAToolkit REQUIRED)
set(CULIBRARIES CUDA::cudart_static CUDA::cublas CUDA::cufft CUDA::nppc CUDA::npps CUDA::curand)

# Pybind11 and Python paths
set(CONDA_ROOT_DIR "D:/miniconda3/envs/Measurement")
set(Python_EXECUTABLE ${CONDA_ROOT_DIR}/python.exe)
set(pybind11_DIR ${CONDA_ROOT_DIR}/Library/share/cmake/pybind11)

# Find Pybind11 and Python
find_package(Python COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED)

# Digitizer Spectrum
set(SPCM_ROOT_DIR c_header)
find_library(SPCM_LIBRARIES spcm_win64_msvcpp ${SPCM_ROOT_DIR})

# Visa
set(VISA_ROOT_DIR "C:/Program Files/IVI Foundation/VISA/Win64")
set(VISA_INCLUDE_DIR ${VISA_ROOT_DIR}/Include)
set(VISA_LIB_DIR ${VISA_ROOT_DIR}/Lib_x64/msc)
find_library(VISA_LIBRARY visa64 ${VISA_LIB_DIR})

# Gather all .cpp, .h, .cu and .cuh files
file(GLOB CPP_SOURCES *.cpp)
file(GLOB CPP_HEADERS *.h)
file(GLOB CUDA_SOURCES *.cu)
file(GLOB CUDA_HEADERS *.cuh)

# Link the libraries to your target (assuming your target is named 'MyCudaProject')
# add_executable(${PROJECT_NAME} main.cpp digitizer.cpp dsp.cu yokogawa_gs210.cpp measurement.cu)
# target_include_directories(${PROJECT_NAME} PUBLIC ${SPCM_ROOT_DIR} ${VISA_INCLUDE_DIR})
# target_link_libraries(${PROJECT_NAME} PRIVATE ${CULIBRARIES} ${SPCM_LIBRARIES} ${VISA_LIBRARY})
# target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")


pybind11_add_module(${PROJECT_NAME} binding.cpp digitizer.cpp dsp.cu yokogawa_gs210.cpp measurement.cu )
target_include_directories(${PROJECT_NAME} PUBLIC ${SPCM_ROOT_DIR} ${VISA_INCLUDE_DIR} ${Python_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CULIBRARIES} ${SPCM_LIBRARIES} ${VISA_LIBRARY} ${Python_LIBRARIES})
target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")
