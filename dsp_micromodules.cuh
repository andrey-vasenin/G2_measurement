#pragma once
#ifndef DSP_MICROMODULES_H
#define DSP_MICROMODULES_H
#include <vector>
#include <any>
#include "dsp_types.cuh"

class micromodule
{
    // virtual micromodule(int len, std::shared_ptr<cufftHandle[]> cufft_plans, std::shared_ptr<cudaStream_t[]> cuda_streams) = default;
    virtual void compute(gpuvec_c &data, gpuvec_c &noise) = 0;
    virtual std::vector<std::any> getResult() = 0;
};
#endif // DSP_MICROMODULES_H