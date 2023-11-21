//
// Created by andrei on 3/27/21.
//

#include "dsp.cuh"
#include "dsp_functors.cuh"
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <npp.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <npps.h>
#include <complex>
#include <cublas_v2.h>
#include <cmath>
#include <numeric>
#include "strided_range.cuh"
#include "tiled_range.cuh"
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/iterator/constant_iterator.h>
#include "npp_status_check.h"



inline void check_cufft_error(cufftResult cufft_err, std::string &&msg)
{
#ifdef _DEBUG

    if (cufft_err != CUFFT_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_cublas_error(cublasStatus_t err, std::string &&msg)
{
#ifdef _DEBUG

    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_npp_error(NppStatus err, std::string &&msg)
{
#ifdef _DEBUG
    if (err != NPP_SUCCESS)
        throw std::runtime_error(NppStatusToString(err) + "; " + msg);
#endif // NDEBUG
}

template <typename T>
inline void print_vector(thrust::device_vector<T> &vec, int n)
{
    cudaDeviceSynchronize();
    thrust::copy(vec.begin(), vec.begin() + n, std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

inline void print_gpu_buff(gpubuf vec, int n)
{
    cudaDeviceSynchronize();
    thrust::copy(vec.begin(), vec.begin() + n, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

// DSP constructor
dsp::dsp(size_t len, uint64_t n, double part,
         double samplerate, int second_oversampling) : trace_length{static_cast<size_t>(std::round((double)len * part))}, // Length of a signal or noise trace
                                                       batch_size{n},                                                     // Number of segments in a buffer (same: number of traces in data)
                                                       total_length{batch_size * trace_length},
                                                       oversampling{second_oversampling},
                                                       resampled_trace_length{trace_length / oversampling},
                                                       resampled_total_length{total_length / oversampling},
                                                       out_size{trace_length * trace_length},
                                                       trace1_start{0},       // Start of the signal data
                                                       trace2_start{len / 2}, // Start of the noise data
                                                       pitch{len}             // Segment length in a buffer

{
    downconversion_coeffs.resize(total_length, tcf(0.f));
    firwin.resize(total_length, tcf(0.f)); // GPU memory for the filtering window
    subtraction_trace1.resize(total_length, tcf(0.f));
    subtraction_trace2.resize(total_length, tcf(0.f));
    corr_firwin[0].resize(total_length, tcf(0.f));
    corr_firwin[1].resize(total_length, tcf(0.f));
    
    // Streams
    for (int i = 0; i < num_streams; i++)
    {
        // Create streams for parallel data processing
        handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        check_npp_error(nppGetStreamContext(&streamContexts[i]), "Npp Error GetStreamContext");
        streamContexts[i].hStream = streams[i];

        // Allocate arrays on GPU for every stream
        gpu_data_buf[i].resize(2 * total_length, '\0');

        // Allocate arrays on GPU for every channel of digitizer
        data1[i].resize(total_length, tcf(0.f));
        data2[i].resize(total_length, tcf(0.f));

        subtraction_data1[i].resize(total_length, tcf(0.f));
        subtraction_data2[i].resize(total_length, tcf(0.f));

        data_for_correlation1[i].resize(total_length, tcf(0.f));
        data_for_correlation2[i].resize(total_length, tcf(0.f));

        g2_out[i].resize(out_size, tcf(0.f));
        g2_out_filtered[i].resize(out_size), tcf(0.f);
        
        // Initialize cuFFT plans
        check_cufft_error(cufftPlan1d(&plans[i], static_cast<int>(trace_length),
                                      CUFFT_C2C, static_cast<int>(batch_size)),
                          "Error initializing cuFFT plan\n");
        // Assign streams to cuFFT plans
        check_cufft_error(cufftSetStream(plans[i], streams[i]),
                          "Error assigning a stream to a cuFFT plan\n");
        // Initialize cuBLAS
        check_cublas_error(cublasCreate(&cublas_handles[i]),
                           "Error initializing a cuBLAS handle\n");
        check_cublas_error(cublasCreate(&cublas_handles2[i]),
                           "Error initializing a cuBLAS handle\n");
        // Assign streams to cuBLAS handles
        check_cublas_error(cublasSetStream(cublas_handles[i], streams[i]),
                           "Error assigning a stream to a cuBLAS handle\n");
        check_cublas_error(cublasSetStream(cublas_handles2[i], streams[i]),
                           "Error assigning a stream to a cuBLAS handle\n");
    }
}

// DSP destructor
dsp::~dsp()
{
    deleteBuffer();
    deleteInterBuffer();
    for (int i = 0; i < num_streams; i++)
    {
        // Destroy cuBLAS
        cublasDestroy(cublas_handles[i]);
        cublasDestroy(cublas_handles2[i]);

        // Destroy cuFFT plans
        cufftDestroy(plans[i]);

        // Destroy GPU streams
        handleError(cudaStreamDestroy(streams[i]));
    }
}

// Set filtering window for digital processing
void dsp::setFirwin(float cutoff_l, float cutoff_r, int oversampling)
{
    makeFilterWindow(cutoff_l, cutoff_r, firwin, oversampling);
}

// Set filtering window for digital processing before G2 calculations
void dsp::setCorrelationFirwin(float cutoff_1[2], float cutoff_2[2], int oversampling)
{
    makeFilterWindow(cutoff_1[0], cutoff_1[1], corr_firwin[0], oversampling);
    makeFilterWindow(cutoff_2[0], cutoff_2[1], corr_firwin[1], oversampling);
}

// Creates a rectangular window with specified cutoff frequencies for the further usage in a filter
void dsp::makeFilterWindow(float cutoff_l, float cutoff_r, gpuvec_c &window, int oversampling)
{
    using namespace std::complex_literals;
    hostvec_c hFirwin(total_length);
    float fs = 1250.f / (float)oversampling;
    int l_idx = (int)std::roundf((float)trace_length / fs * cutoff_l);
    int r_idx = (int)std::roundf((float)trace_length / fs * cutoff_r);
    for (int i = 0; i < total_length; i++)
    {
        int j = i % trace_length;
        hFirwin[i] = ((j < l_idx) || (j > r_idx)) ? 0if : 1.0f + 0if;
    }
    window = hFirwin;
}

// Error handler
void dsp::handleError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::string name = cudaGetErrorName(err);
        std::string text = cudaGetErrorString(err);
        throw std::runtime_error(name + ": " + text);
    }
}

void dsp::createBuffer(size_t size)
{
    this->handleError(cudaMallocHost((void **)&buffer, size));
}

void dsp::deleteBuffer()
{
    if (buffer != nullptr)
        this->handleError(cudaFreeHost(buffer));
}

void dsp::createInterBuffer(size_t size)
{
    this->handleError(cudaMallocHost((void **)&inter_buffer, size));
}

void dsp::deleteInterBuffer()
{
    if (inter_buffer != nullptr)
        this->handleError(cudaFreeHost(inter_buffer));
}

hostbuf dsp::getBuffer()
{
    return buffer;
}

void dsp::setIntermediateFrequency(float frequency, int oversampling)
{
    const float pi = std::acos(-1.f);
    float ovs = static_cast<float>(oversampling);
    hostvec_c hDownConv(total_length);
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
                     [=] __host__(int i) -> tcf
                     {
                         float t = 0.8f * ovs * static_cast<float>(i % trace_length);
                         return thrust::exp(tcf(0.f, -2.f * pi * frequency * t));
                     });
    downconversion_coeffs = hDownConv;
}

void dsp::downconvert(gpuvec_c &data, int stream_num)
{
    // thrust::transform(thrust::cuda::par_nosync.on(stream), data.begin(), data.end(), downconversion_coeffs.begin(), data.begin(), downconv_functor());
    Npp32fc *src = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(data.data()));
    const Npp32fc *coef = reinterpret_cast<const Npp32fc *>(thrust::raw_pointer_cast(downconversion_coeffs.data()));
    auto status = nppsMul_32fc_I_Ctx(coef, src, static_cast<int>(data.size()), streamContexts[stream_num]);
    check_npp_error(status, "Error with downconversion");
}

void dsp::setDownConversionCalibrationParameters(float r, float phi,
                                                 float offset_i, float offset_q)
{
    a_qi = std::tan(phi);
    a_qq = 1 / (r * std::cos(phi));
    c_i = offset_i;
    c_q = offset_q;
}

// Applies down-conversion calibration to traces
void dsp::applyDownConversionCalibration(gpuvec_c &data, cudaStream_t &stream)
{
    auto sync_exec_policy = thrust::cuda::par_nosync.on(stream);
    thrust::for_each(sync_exec_policy, data.begin(), data.end(), calibration_functor(a_qi, a_qq, c_i, c_q));
}
// Fills with zeros the arrays for results output in the GPU memory
void dsp::resetOutput()
{
    for (int i = 0; i < num_streams; i++)
    {
        thrust::fill(subtraction_data1[i].begin(), subtraction_data1[i].end(), tcf(0));
        thrust::fill(subtraction_data2[i].begin(), subtraction_data2[i].end(), tcf(0));
        thrust::fill(g2_out[i].begin(), g2_out[i].end(), tcf(0));
        thrust::fill(g2_out_filtered[i].begin(), g2_out[i].end(), tcf(0));
        thrust::fill(data_for_correlation1[i].begin(), data_for_correlation1[i].begin(), tcf(0));
        thrust::fill(data_for_correlation2[i].begin(), data_for_correlation2[i].begin(), tcf(0));
    }
}

void dsp::compute(const hostbuf buffer_ptr)
{
    const int stream_num = semaphore;
    switchStream();
    divideDataFromDifferentChannels(buffer_ptr, gpu_data_buf[stream_num], 0, stream_num);
    convertDataToMillivolts(data1[stream_num], gpu_data_buf[stream_num], streams[stream_num]);
    applyDownConversionCalibration(data1[stream_num], streams[stream_num]);
    applyFilter(data1[stream_num], firwin, stream_num);
    downconvert(data1[stream_num], stream_num);
    subtractDataFromOutput(subtraction_trace1, data1[stream_num], stream_num);
    addDataToOutput(data1[stream_num], subtraction_data1[stream_num], stream_num);

    divideDataFromDifferentChannels(buffer_ptr, gpu_data_buf[stream_num], 1, stream_num);
    convertDataToMillivolts(data2[stream_num], gpu_data_buf[stream_num], streams[stream_num]);
    applyDownConversionCalibration(data2[stream_num], streams[stream_num]);
    applyFilter(data2[stream_num], firwin, stream_num);
    downconvert(data2[stream_num], stream_num);
    subtractDataFromOutput(subtraction_trace2, data2[stream_num], stream_num);
    addDataToOutput(data2[stream_num], subtraction_data2[stream_num], stream_num);

    // Example of calculateG2 usage
    data_for_correlation1[stream_num] = data1[stream_num];
    applyFilter(data_for_correlation1[stream_num], corr_firwin[0], stream_num);
    data_for_correlation2[stream_num] = data2[stream_num];
    applyFilter(data_for_correlation2[stream_num], corr_firwin[1], stream_num);

    calculateG2(data_for_correlation1[stream_num], data_for_correlation2[stream_num], g2_out_filtered[stream_num],
                streams[stream_num], cublas_handles[stream_num]);
    calculateG2(data1[stream_num], data2[stream_num], g2_out[stream_num],
                streams[stream_num], cublas_handles[stream_num]);
}

// This function uploads data from the specified section of a buffer array to the GPU memory
void dsp::divideDataFromDifferentChannels(const hostbuf buffer_ptr,
                                          gpubuf &dst, size_t offset, int stream_num)
{
    size_t width = 2 * sizeof(int8_t);
    size_t src_pitch = 2 * num_channels * sizeof(int8_t);
    size_t dst_pitch = width;
    size_t shift = 2 * num_channels * offset;
    size_t height = batch_size * trace_length;
    handleError(cudaMemcpy2DAsync(thrust::raw_pointer_cast(dst.data()), dst_pitch,
                                  static_cast<const void *>(buffer_ptr + shift), src_pitch, width, height,
                                  cudaMemcpyHostToDevice, streams[stream_num]));
}

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMillivolts(gpuvec_c &data, const gpubuf &gpu_buf, const cudaStream_t &stream)
{
    using iter = gpubuf::const_iterator;
    strided_range<iter> channelI(gpu_buf.begin(), gpu_buf.end(), 2);
    strided_range<iter> channelQ(gpu_buf.begin() + 1, gpu_buf.end(), 2);
    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      channelI.begin(), channelI.end(), channelQ.begin(), data.begin(), millivolts_functor(scale));
}

// Applies the filter with the specified window to the data using FFT convolution
void dsp::applyFilter(gpuvec_c &data, const gpuvec_c &window, int stream_num)
{
    // Step 1. Take FFT of each segment
    cufftComplex *cufft_data = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(data.data()));
    auto cufftstat = cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_FORWARD);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 2. Multiply each segment by a window
    thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
                      data.begin(), data.end(), window.begin(), data.begin(), thrust::multiplies<tcf>());
    // Step 3. Take inverse FFT of each segment
    cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_INVERSE);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 4. Normalize the FFT for the output to equal the input
    thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
                      data.begin(), data.end(), thrust::constant_iterator<tcf>(1.f / static_cast<float>(trace_length)),
                      data.begin(), thrust::multiplies<tcf>());
}

// Sums newly processed data with previous data for averaging
void dsp::addDataToOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num)
{
    const Npp32fc *src = reinterpret_cast<const Npp32fc *>(thrust::raw_pointer_cast(data.data()));
    Npp32fc *dst = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(output.data()));
    auto status = nppsAdd_32fc_I_Ctx(src, dst, data.size(), streamContexts[stream_num]);
    check_npp_error(status, "Error adding two vectors");
}

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num)
{
    const Npp32fc *src = reinterpret_cast<const Npp32fc *>(thrust::raw_pointer_cast(data.data()));
    Npp32fc *dst = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(output.data()));
    auto status = nppsSub_32fc_I_Ctx(src, dst, data.size(), streamContexts[stream_num]);
    check_npp_error(status, "Error subtracting two vectors");
    /*thrust::transform(thrust::cuda::par_nosync.on(stream), output.begin(), output.end(), data.begin(),
        output.begin(), thrust::minus<tcf>());*/
}

void dsp::resample(const gpuvec_c &traces, gpuvec_c &resampled_traces, const cudaStream_t &stream)
{
    using iter = gpuvec_c::const_iterator;
    switch (oversampling)
    {
    case 1:
        thrust::copy(thrust::cuda::par_nosync.on(stream), traces.begin(), traces.end(), resampled_traces.begin());
        break;
    case 2:
    {
        strided_range<iter> t1(traces.begin(), traces.end(), oversampling);
        strided_range<iter> t2(traces.begin() + 1, traces.end(), oversampling);
        auto beginning = thrust::make_zip_iterator(t1.begin(), t2.begin());
        auto end = thrust::make_zip_iterator(t1.end(), t2.end());
        thrust::transform(thrust::cuda::par_nosync.on(stream), beginning, end, resampled_traces.begin(),
                          thrust::make_zip_function(downsample2_functor()));
        break;
    }
    case 4:
    {
        strided_range<iter> t1(traces.begin(), traces.end(), oversampling);
        strided_range<iter> t2(traces.begin() + 1, traces.end(), oversampling);
        strided_range<iter> t3(traces.begin() + 2, traces.end(), oversampling);
        strided_range<iter> t4(traces.begin() + 3, traces.end(), oversampling);
        auto beginning = thrust::make_zip_iterator(t1.begin(), t2.begin(), t3.begin(), t4.begin());
        auto end = thrust::make_zip_iterator(t1.end(), t2.end(), t3.end(), t4.end());
        thrust::transform(thrust::cuda::par_nosync.on(stream), beginning, end, resampled_traces.begin(),
                          thrust::make_zip_function(downsample4_functor()));
        break;
    }
    default:
        throw std::runtime_error("Unsupported second oversampling");
    }
}

void dsp::normalize(gpuvec_c &data, float coeff, int stream_num)
{
    Npp32fc x{coeff, 0.f};
    nppsMulC_32fc_I_Ctx(x, reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(data.data())),
                        data.size(), streamContexts[stream_num]);
}

// Calculate second-order correlation function.
// Could be used for calculating correlations between filtered signals.
// For this use dsp::makeFilterWindow and dsp::applyFilter before calling this method.
void dsp::calculateG2(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &output, const cudaStream_t &stream, cublasHandle_t &handle)
{
    gpuvec_c crossPower(total_length, tcf(0)); // contains cross-channel power of signal
    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      data_1.begin(), data_1.end(), data_2.begin(), crossPower.begin(), cross_power_functor());
    // Calculating G2 as two-time cross power correlation
    using namespace std::string_literals;
    float alpha_data = 1;
    float beta = 1;
    auto cublas_status = cublasCsyrk(handle,
                                     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                     reinterpret_cast<cuComplex *>(&alpha_data),
                                     reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(crossPower.data())), trace_length,
                                     reinterpret_cast<cuComplex *>(&beta),
                                     reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output.data())), trace_length);
    // Check for errors
    check_cublas_error(cublas_status,
                       "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
}

template <typename T>
thrust::host_vector<T> dsp::getCumulativeTrace(const thrust::device_vector<T> *traces)
{
    handleError(cudaDeviceSynchronize());
    thrust::device_vector<T> tmp(traces->size(), T(0));
    for (int i = 0; i < num_streams; i++)
        thrust::transform(traces[i].begin(), traces[i].end(), tmp.begin(), tmp.begin(), thrust::plus<T>());
    size_t N = traces->size() / batch_size;
    thrust::host_vector<T> host_trace(N);
    using iter = typename thrust::device_vector<T>::iterator;
    for (size_t j = 0; j < N; ++j)
    {
        strided_range<iter> tmp_iter(tmp.begin() + j, tmp.end(), N);
        T el = thrust::reduce(tmp_iter.begin(), tmp_iter.end(), T(0), thrust::plus<T>());
        host_trace[j] = el / T(batch_size);
    }
    return host_trace;
}

gpuvec_c dsp::getCumulativeCorrelator(gpuvec_c g_out[4])
{
    gpuvec_c c(g_out[0].size(), tcf(0));
    this->handleError(cudaDeviceSynchronize());
    for (int i = 0; i < num_streams; i++)
        thrust::transform(g_out[i].begin(), g_out[i].end(), c.begin(), c.begin(), thrust::plus<tcf>());
    return c;
}

void dsp::getG2FullResults(hostvec_c &result)
{
    result = getCumulativeCorrelator(g2_out);
}

void dsp::getG2FilteredResults(hostvec_c &result)
{
    result = getCumulativeCorrelator(g2_out_filtered);
}

std::vector<hostvec_c> dsp::getCumulativeSubtrData()
{
    std::vector<hostvec_c> subtr_data;
    subtr_data.push_back(getCumulativeTrace(subtraction_data1));
    subtr_data.push_back(getCumulativeTrace(subtraction_data2));
    return subtr_data;
}

// Returns the useful length of the data in a segment
// (trace is assumed complex valued)
int dsp::getTraceLength()
{
    return trace_length;
}

// Returns the total length of the data comprised of several segments
// (trace is assumed complex valued)
int dsp::getTotalLength()
{
    return total_length;
}

int dsp::getOutSize()
{
    return out_size;
}

void dsp::setAmplitude(int ampl)
{
    scale = static_cast<float>(ampl) / 128.f;
}

void dsp::setSubtractionTrace(hostvec_c trace[num_channels])
{
    subtraction_trace1 = trace[0];
    subtraction_trace2 = trace[1];
}

void dsp::getSubtractionTrace(std::vector<hostvec_c> &trace)
{
    trace.push_back(subtraction_trace1);
    trace.push_back(subtraction_trace2);
}

void dsp::resetSubtractionTrace()
{
    thrust::fill(subtraction_trace1.begin(), subtraction_trace1.end(), tcf(0));
    thrust::fill(subtraction_trace2.begin(), subtraction_trace2.end(), tcf(0));
}