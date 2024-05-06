#pragma once
#ifndef DSP_TYPES_H
#define PSD_TYPES_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

typedef thrust::complex<float> tcf;
typedef thrust::device_vector<float> gpuvec;
typedef thrust::host_vector<float> hostvec;
typedef thrust::device_vector<tcf> gpuvec_c;
typedef thrust::host_vector<tcf> hostvec_c;
typedef thrust::device_vector<char2> gpubuf;
typedef int8_t *hostbuf;
typedef std::vector<float> stdvec;
typedef std::vector<std::complex<float>> stdvec_c;
#endif // DSP_TYPES_H