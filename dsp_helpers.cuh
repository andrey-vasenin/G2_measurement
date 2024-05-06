#pragma once
#ifndef DSP_HELPERS_H
#define DSP_HELPERS_H
#include <string>
#include <cufft.h>
#include <cublas_v2.h>
#include <nppcore.h>
#include <thrust/device_vector.h>
// #include "npp_status_check.h"

#define _DEBUG1

template <typename T>
inline T *get(thrust::device_vector<T> vec)
{
    return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline Npp32fc *to_Npp32fc_p(T *v)
{
    return reinterpret_cast<Npp32fc *>(v);
}

template <typename T>
inline Npp32f *to_Npp32f_p(T *v)
{
    return reinterpret_cast<Npp32f *>(v);
}

inline void check_cufft_error(cufftResult cufft_err, std::string &&msg)
{
#ifdef _DEBUG1

    if (cufft_err != CUFFT_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_cublas_error(cublasStatus_t err, std::string &&msg)
{
#ifdef _DEBUG1

    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_npp_error(NppStatus err, std::string &&msg)
{
// #ifdef _DEBUG1
//     if (err != NPP_SUCCESS)
//         throw std::runtime_error(NppStatusToString(err) + "; " + msg);
// #endif // NDEBUG
}

template <typename T>
inline void print_vector(thrust::device_vector<T> &vec, int n)
{
    cudaDeviceSynchronize();
    thrust::copy(vec.begin(), vec.begin() + n, std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}
#endif // DSP_HELPERS_H