﻿//
// Created by andrei on 3/27/21.
//

#ifndef CPPMEASUREMENT_DSP_CUH
#define CPPMEASUREMENT_DSP_CUH

#include <nppdefs.h>
#include <vector>
#include <complex>
#include <cufft.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>

const int num_streams = 4;
const int cal_mat_size = 16;
const int cal_mat_side = 4;

typedef thrust::complex<float> tcf;
typedef thrust::device_vector<float> gpuvec;
typedef thrust::host_vector<float> hostvec;
typedef thrust::device_vector<tcf> gpuvec_c;
typedef thrust::host_vector<tcf> hostvec_c;
typedef thrust::device_vector<char> gpubuf;
typedef thrust::host_vector<int8_t, thrust::mr::stateless_resource_allocator<int8_t,
    thrust::system::cuda::universal_host_pinned_memory_resource> > hostbuf;
typedef hostbuf::iterator hostbuf_iter_t;
typedef std::vector<float> stdvec;
typedef std::vector<std::complex<float>> stdvec_c;

template <typename T>
inline T *get(thrust::device_vector<T> vec)
{
    return thrust::raw_pointer_cast(vec.data());
}

template <typename T>
inline Npp32fc* to_Npp32fc_p(T* v)
{
    return reinterpret_cast<Npp32fc*>(v);
}

template <typename T>
inline Npp32f* to_Npp32f_p(T* v)
{
    return reinterpret_cast<Npp32f*>(v);
}

class dsp
{
    /* Pointer */
    hostbuf buffer;

    /* Pointers to arrays with data */
    gpubuf gpu_buf[num_streams];  // buffers for loading data
    gpubuf gpu_buf2[num_streams]; // buffers for loading data
    gpuvec_c data[num_streams];
    gpuvec_c data_calibrated[num_streams];
    gpuvec_c noise[num_streams];
    gpuvec_c noise_calibrated[num_streams];

    gpuvec power[num_streams];   // arrays for storage of average power
    gpuvec_c field[num_streams]; // arrays for storage of average field
    gpuvec_c out[num_streams];
    gpuvec spectrum[num_streams];

    uint32_t cnt = 0;

    /* Filtering window */
    gpuvec_c firwin;

    /* Subtraction trace */
    gpuvec_c subtraction_trace;

    /* Downconversion coefficients */
    gpuvec_c downconversion_coeffs;

private:
    /* Useful variables */
    size_t trace_length; // for keeping the length of a trace
    size_t trace1_start, trace2_start, pitch;
    size_t batch_size;   // for keeping the number of segments in data array  // was uint64_t
    size_t total_length; // batch_size * trace_length
    size_t out_size;
    int semaphore = 0;                           // for selecting the current stream
    float scale = 500.f / 128.f; // for conversion into mV

    /* Streams' arrays */
    cudaStream_t streams[num_streams];
    NppStreamContext streamContexts[num_streams];

    /* cuFFT required variables */
    cufftHandle plans[num_streams];

    /* cuBLAS required variables */
    cublasHandle_t cublas_handles[num_streams];
    cublasHandle_t cublas_handles2[num_streams];

    /* Down-conversion calibration variables */
    gpuvec A_gpu;
    gpuvec_c C_gpu;
    int batch_count;
    float alpha = 1.f;
    float beta = 0.f;

    float a_ii, a_qi, a_qq, c_i, c_q;

public:
    dsp(size_t len, uint64_t n, double part);

    ~dsp();

    int getTraceLength();

    int getTotalLength();

    int getOutSize();

    void setFirwin(float cutoff_l, float cutoff_r, int oversampling = 1);

    void resetOutput();

    uint32_t getCounter() { return cnt; }

    void compute(const int8_t* buffer_ptr);

    void getCumulativePower(hostvec &result);

    void getCumulativeSpectrum(hostvec& result);

    void getCumulativeField(hostvec_c &result);

    void getCorrelator(hostvec_c &result);

    void setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q);

    void setSubtractionTrace(hostvec_c &trace);

    void getSubtractionTrace(hostvec_c &trace);

    void resetSubtractionTrace();

    void createBuffer(size_t size);

    void setIntermediateFrequency(float frequency, int oversampling);

    int8_t* getBuffer();

    void setAmplitude(int ampl);

protected:
    void handleError(cudaError_t error);

    void switchStream() { semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0; };

    void loadDataToGPUwithPitchAndOffset(const int8_t* buffer_ptr,
        gpubuf & gpu_buf, size_t pitch, size_t offset, int stream_num);

    void convertDataToMillivolts(gpuvec_c& data, gpubuf& gpu_buf, cudaStream_t& stream);

    void downconvert(gpuvec_c data, cudaStream_t& stream);

    void applyDownConversionCalibration(gpuvec_c &data, gpuvec_c &data_calibrated, cudaStream_t& stream);

    void addDataToOutput(gpuvec_c& data, gpuvec_c& output, cudaStream_t& stream);

    void subtractDataFromOutput(gpuvec_c& data, gpuvec_c& output, cudaStream_t& stream);

    void applyFilter(gpuvec_c &data, const gpuvec_c &window, int stream_num);

    void calculateField(gpuvec_c& data, gpuvec_c& noise, gpuvec_c& output, cudaStream_t& stream);

    void calculatePower(gpuvec_c& data, gpuvec_c& noise, gpuvec& output, cudaStream_t& stream);

    void calculateG1(gpuvec_c& data, gpuvec_c& noise, gpuvec_c& output, cublasHandle_t &handle);

    void calculateTriplets(gpuvec_c& data, gpuvec_c& noise, gpuvec& output, int stream_num);
};

#endif // CPPMEASUREMENT_DSP_CUH
