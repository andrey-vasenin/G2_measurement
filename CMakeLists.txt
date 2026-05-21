cmake_minimum_required(VERSION 3.20)

project(AverageField LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

find_package(CUDAToolkit REQUIRED)
set(CULIBRARIES CUDA::cudart_static CUDA::cublas CUDA::cufft CUDA::nppc CUDA::npps CUDA::curand)

set(CONDA_ROOT_DIR "" CACHE PATH "Optional conda environment root, for example C:/Users/Qop/miniconda3/envs/qom")
if(CONDA_ROOT_DIR AND NOT Python_EXECUTABLE)
    set(Python_EXECUTABLE "${CONDA_ROOT_DIR}/python.exe" CACHE FILEPATH "Python executable" FORCE)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_pybind11_cmake_dir)
        set(pybind11_DIR "${_pybind11_cmake_dir}" CACHE PATH "pybind11 CMake package directory" FORCE)
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

set(SPCM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_header" CACHE PATH "Spectrum SDK c_header directory")
foreach(_spcm_header dlltyp.h regs.h spcerr.h spcm_drv.h)
    if(NOT EXISTS "${SPCM_ROOT_DIR}/${_spcm_header}")
        message(FATAL_ERROR "Missing Spectrum header ${_spcm_header} in SPCM_ROOT_DIR=${SPCM_ROOT_DIR}")
    endif()
endforeach()
find_library(SPCM_LIBRARY NAMES spcm_win64_msvcpp PATHS "${SPCM_ROOT_DIR}" REQUIRED)

option(AVERAGEFIELD_WITH_VISA "Link VISA for legacy instrument-control builds" ON)
if(AVERAGEFIELD_WITH_VISA)
    set(VISA_ROOT_DIR "C:/Program Files/IVI Foundation/VISA/Win64" CACHE PATH "VISA Win64 root directory")
    set(VISA_INCLUDE_DIR "${VISA_ROOT_DIR}/Include" CACHE PATH "VISA include directory")
    set(VISA_LIB_DIR "${VISA_ROOT_DIR}/Lib_x64/msc" CACHE PATH "VISA library directory")
    find_library(VISA_LIBRARY NAMES visa64 PATHS "${VISA_LIB_DIR}" REQUIRED)
endif()

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python NumPy include dirs: ${Python_NumPy_INCLUDE_DIRS}")
message(STATUS "pybind11_DIR: ${pybind11_DIR}")
message(STATUS "SPCM_ROOT_DIR: ${SPCM_ROOT_DIR}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

pybind11_add_module(${PROJECT_NAME} binding.cpp digitizer.cpp dsp.cu measurement.cu)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_include_directories(${PROJECT_NAME} PRIVATE "${SPCM_ROOT_DIR}" ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CULIBRARIES} "${SPCM_LIBRARY}")
if(AVERAGEFIELD_WITH_VISA)
    target_include_directories(${PROJECT_NAME} PRIVATE "${VISA_INCLUDE_DIR}")
    target_link_libraries(${PROJECT_NAME} PRIVATE "${VISA_LIBRARY}")
endif()
target_link_options(${PROJECT_NAME} PRIVATE "/NODEFAULTLIB:LIBCMT")

set(AVERAGEFIELD_SMOKE_IMPORT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/smoke_import.py")
set(AVERAGEFIELD_SMOKE_MEASURE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/smoke_measure_test.py")

add_custom_target(smoke_import
    COMMAND "${Python_EXECUTABLE}" "${AVERAGEFIELD_SMOKE_IMPORT_SCRIPT}" --module "$<TARGET_FILE:${PROJECT_NAME}>"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Import-smoke-testing AverageField"
    VERBATIM
)

add_custom_target(smoke_measure_test
    COMMAND "${Python_EXECUTABLE}" "${AVERAGEFIELD_SMOKE_MEASURE_SCRIPT}" --module "$<TARGET_FILE:${PROJECT_NAME}>"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running synthetic AverageField measure_test regression"
    VERBATIM
)

add_custom_target(smoke_all
    COMMAND "${Python_EXECUTABLE}" "${AVERAGEFIELD_SMOKE_IMPORT_SCRIPT}" --module "$<TARGET_FILE:${PROJECT_NAME}>"
    COMMAND "${Python_EXECUTABLE}" "${AVERAGEFIELD_SMOKE_MEASURE_SCRIPT}" --module "$<TARGET_FILE:${PROJECT_NAME}>"
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running all AverageField smoke tests"
    VERBATIM
)

if(BUILD_TESTING)
    add_test(
        NAME averagefield_import
        COMMAND "${Python_EXECUTABLE}" "${AVERAGEFIELD_SMOKE_IMPORT_SCRIPT}" --module "$<TARGET_FILE:${PROJECT_NAME}>"
    )
    set_tests_properties(averagefield_import PROPERTIES
        LABELS "smoke"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(
        NAME averagefield_measure_test
        COMMAND "${Python_EXECUTABLE}" "${AVERAGEFIELD_SMOKE_MEASURE_SCRIPT}" --module "$<TARGET_FILE:${PROJECT_NAME}>"
    )
    set_tests_properties(averagefield_measure_test PROPERTIES
        LABELS "smoke;gpu"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()
