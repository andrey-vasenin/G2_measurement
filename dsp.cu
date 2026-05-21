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
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/iterator/constant_iterator.h>
#include "npp_status_check.h"

// #define _DEBUG1

inline void check_cufft_error(cufftResult cufft_err, std::string&& msg)
{
#ifdef _DEBUG1

    if (cufft_err != CUFFT_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_cublas_error(cublasStatus_t err, std::string&& msg)
{
#ifdef _DEBUG1

    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_npp_error(NppStatus err, std::string&& msg)
{
#ifdef _DEBUG1
    if (err != NPP_SUCCESS)
        throw std::runtime_error(NppStatusToString(err) + "; " + msg);
#endif // NDEBUG
}

template <typename T>
inline void print_vector(thrust::device_vector<T>& vec, int n)
{
    cudaDeviceSynchronize();
    thrust::copy(vec.begin(), vec.begin() + n, std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

// inline void print_gpu_buff(gpubuf vec, int n)
// {
//     cudaDeviceSynchronize();
//     thrust::copy(vec.begin(), vec.begin() + n, std::ostream_iterator<int>(std::cout, " "));
//     std::cout << std::endl;
// }

// DSP constructor
dsp::dsp(size_t len, uint64_t n,
    double samplerate, int second_oversampling, ResultMode mode) : trace_length{ len }, // Length of a signal or noise trace
    batch_size{ n },                                                     // Number of segments in a buffer (same: number of traces in data)
    total_length{ batch_size * trace_length },
    oversampling{ second_oversampling },
    resampled_trace_length{ trace_length / oversampling },
    resampled_total_length{ total_length / oversampling },
    out_size{ resampled_trace_length * resampled_trace_length },
    result_mode{ mode },
    pitch{ len }

{
    downconversion_coeffs.resize(total_length, tcf(0.f));
    corr_downconversion_coeffs1.resize(resampled_total_length, tcf(0.f));
    corr_downconversion_coeffs2.resize(resampled_total_length, tcf(0.f));
    firwin.resize(total_length, tcf(0.f)); // GPU memory for the filtering window
    // center_peak_win.resize(resampled_total_length, tcf(0.f));
    // corr_firwin1.resize(resampled_total_length, tcf(0.f));
    // corr_firwin2.resize(resampled_total_length, tcf(0.f)); 
    subtraction_trace1.resize(resampled_total_length, tcf(0.f));
    subtraction_trace2.resize(resampled_total_length, tcf(0.f));
    tmp1.resize(resampled_trace_length, tcf(0.f));
    tmp2.resize(resampled_trace_length, tcf(0.f));
    tmp_cross.resize(resampled_trace_length, tcf(0.f));
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
    // Allocate arrays on GPU for every stream
    for (int i = 0; i < num_streams; i++)
    {
        gpu_data_buf[i].resize(total_length, char4{ 0,0,0,0 });
        // Create streams for parallel data processing
        handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        handleError(cudaEventCreateWithFlags(&input_copy_done[i], cudaEventDisableTiming));
        // check_npp_error(initNppStreamContext(&streamContexts[i], streams[i]), "Npp Error GetStreamContext");
        streamContexts[i].nCudaDeviceId = device_id;
        streamContexts[i].nMultiProcessorCount = prop.multiProcessorCount;
        streamContexts[i].nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        streamContexts[i].nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
        streamContexts[i].nSharedMemPerBlock = prop.sharedMemPerBlock;
        streamContexts[i].nCudaDevAttrComputeCapabilityMajor = major;
        streamContexts[i].nCudaDevAttrComputeCapabilityMinor = minor;
        streamContexts[i].hStream = streams[i];
        cudaStreamGetFlags(streams[i], &streamContexts[i].nStreamFlags);


        // Allocate arrays on GPU for every channel of digitizer
        data1[i].resize(total_length, tcf(0.f));
        data2[i].resize(total_length, tcf(0.f));
        data1_resampled[i].resize(resampled_total_length, tcf(0.f));
        data2_resampled[i].resize(resampled_total_length, tcf(0.f));

        subtraction_data1[i].resize(resampled_total_length, tcf(0.f));
        subtraction_data2[i].resize(resampled_total_length, tcf(0.f));

        if (hasG1())
        {
            data2_resampled_conj[i].resize(resampled_total_length, tcf(0.f));
            g1[i].resize(out_size, tcf(0.f));
        }

        if (hasAllCorrelators())
        {
            data1_resampled_conj[i].resize(resampled_total_length, tcf(0.f));
            g1_annihilation[i].resize(out_size, tcf(0.f));
            g1_creation[i].resize(out_size, tcf(0.f));
            g1_reordered[i].resize(out_size, tcf(0.f));
            cross_power[i].resize(resampled_total_length, tcf(0.f));
            cross_spectrum[i].resize(resampled_total_length, tcf(0.f));
        }

        // data_for_correlation1[i].resize(resampled_total_length, tcf(0.f));
        // data_for_correlation2[i].resize(resampled_total_length, tcf(0.f));

        // data_without_central_peak1[i].resize(resampled_total_length, tcf(0.f));
        // data_without_central_peak2[i].resize(resampled_total_length, tcf(0.f));

        // trace_out[i].resize(resampled_total_length, tcf(0.f));

        // g1_filt[i].resize(out_size, tcf(0.f));
        // g2_out[i].resize(out_size, tcf(0.f));
        // g2_out_cross_segment[i].resize(out_size, tcf(0.f));
        // g2_out_filtered[i].resize(out_size, tcf(0.f));
        // g2_out_filtered_cross_segment[i].resize(out_size, tcf(0.f));
        // cross_power_short[i].resize(resampled_total_length - resampled_trace_length, tcf(0.f));
        // power1[i].resize(resampled_total_length, tcf(0.f));
        // power2[i].resize(resampled_total_length, tcf(0.f));
        // power_short[i].resize(resampled_total_length - resampled_trace_length, tcf(0.f));

        // Initialize cuFFT plans
        check_cufft_error(cufftPlan1d(&plans[i], static_cast<int>(trace_length),
            CUFFT_C2C, static_cast<int>(batch_size)),
            "Error initializing cuFFT plan\n");
        // Assign streams to cuFFT plans
        check_cufft_error(cufftSetStream(plans[i], streams[i]),
            "Error assigning a stream to a cuFFT plan\n");
        if (hasAllCorrelators())
        {
            check_cufft_error(cufftPlan1d(&corr_plans[i], static_cast<int>(resampled_trace_length),
                CUFFT_C2C, static_cast<int>(batch_size)),
                "Error initializing cuFFT plan\n");
            check_cufft_error(cufftSetStream(corr_plans[i], streams[i]),
                "Error assigning a stream to a cuFFT plan\n");
        }
        if (hasG1())
        {
            // Initialize cuBLAS
            check_cublas_error(cublasCreate(&cublas_handles[i]),
                "Error initializing a cuBLAS handle\n");
            // Assign streams to cuBLAS handles
            check_cublas_error(cublasSetStream(cublas_handles[i], streams[i]),
                "Error assigning a stream to a cuBLAS handle\n");
        }
    }

    // Scalar reduction accumulators (avoid per-call allocations in getS21)
    this->handleError(cudaMalloc(reinterpret_cast<void**>(&s21_sum1), sizeof(float2)));
    this->handleError(cudaMalloc(reinterpret_cast<void**>(&s21_sum2), sizeof(float2)));
}

// DSP destructor
dsp::~dsp()
{
    deleteBuffer();

    if (s21_sum1 != nullptr)
        cudaFree(s21_sum1);
    if (s21_sum2 != nullptr)
        cudaFree(s21_sum2);

    for (int i = 0; i < num_streams; i++)
    {
        // Destroy cuBLAS
        if (hasG1())
            cublasDestroy(cublas_handles[i]);

        // Destroy cuFFT plans
        cufftDestroy(plans[i]);
        if (hasAllCorrelators())
            cufftDestroy(corr_plans[i]);

        // Destroy GPU streams
        handleError(cudaEventDestroy(input_copy_done[i]));
        handleError(cudaStreamDestroy(streams[i]));
    }
}

// Set filtering window for digital processing
void dsp::setFirwin(float cutoff_l, float cutoff_r, int dig_oversampling)
{
    makeFilterWindow(cutoff_l, cutoff_r, firwin, trace_length, total_length, dig_oversampling);
}

void dsp::setFirwin(hostvec_c window)
{
    firwin = window;
}

void dsp::setCentralPeakWin(float cutoff_l, float cutoff_r, int dig_oversampling)
{
    makeFilterWindow(cutoff_l, cutoff_r, center_peak_win, resampled_trace_length, resampled_total_length, dig_oversampling * oversampling);
}

void dsp::setCentralPeakWin(hostvec_c window)
{
    center_peak_win = window;
}

// Set filtering window for digital processing before G2 calculations
void dsp::setCorrelationFirwin(std::pair<float, float> cutoff_1, std::pair<float, float> cutoff_2, int dig_oversampling)
{
    makeFilterWindow(cutoff_1.first, cutoff_1.second, corr_firwin1, resampled_trace_length, resampled_total_length, dig_oversampling * oversampling);
    makeFilterWindow(cutoff_2.first, cutoff_2.second, corr_firwin2, resampled_trace_length, resampled_total_length, dig_oversampling * oversampling);
}

void dsp::setCorrelationFirwin(hostvec_c window1, hostvec_c window2)
{
    corr_firwin1 = window1;
    corr_firwin2 = window2;
}

// Creates a rectangular window with specified cutoff frequencies for the further usage in a filter
// Frequencies in MHz
void dsp::makeFilterWindow(float cutoff_l, float cutoff_r, gpuvec_c& window, size_t trace_len, size_t total_len, int oversamp)
{
    using namespace std::complex_literals;
    hostvec_c hFirwin(total_len);
    float fs = 1250.f / (float)oversamp;
    int l_idx = (int)std::roundf((float)trace_len / fs * cutoff_l);
    int r_idx = (int)std::roundf((float)trace_len / fs * cutoff_r);
    for (int i = 0; i < total_len; i++)
    {
        int j = i % trace_len;
        j = (j > trace_len / 2) ? j - int(trace_len) : j; // according to FFT frequency order
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
    this->handleError(cudaMallocHost((void**)&buffer, size));
}

void dsp::deleteBuffer()
{
    if (buffer != nullptr)
        this->handleError(cudaFreeHost(buffer));
}

hostbuf dsp::getBuffer()
{
    return buffer;
}

void dsp::setIntermediateFrequency(float frequency, int oversampling)
{
    const float pi = std::acos(-1.f);
    float ovs = static_cast<float>(oversampling);
    const int trace_len = static_cast<int>(trace_length);
    hostvec_c hDownConv(total_length);
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
        [pi, ovs, frequency, trace_len] __host__(int i) -> tcf
    {
        float t = 0.8f * ovs * static_cast<float>(i % trace_len);
        return thrust::exp(tcf(0.f, -2.f * pi * frequency * t));
    });
    downconversion_coeffs = hDownConv;
}

void dsp::setCorrDowncovertCoeffs(float freq1, float freq2, int oversampling)
{
    const float pi = std::acos(-1.f);
    float ovs = static_cast<float>(oversampling);
    const int resampled_trace_len = static_cast<int>(resampled_trace_length);
    hostvec_c hDownConv(resampled_total_length);
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
        [pi, ovs, freq1, resampled_trace_len] __host__(int i) -> tcf
    {
        float t = 0.8f * ovs * static_cast<float>(i % resampled_trace_len);
        return thrust::exp(tcf(0.f, -2.f * pi * freq1 * t));
    });
    corr_downconversion_coeffs1 = hDownConv;

    thrust::fill(hDownConv.begin(), hDownConv.end(), tcf(0.f));
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
        [pi, ovs, freq2, resampled_trace_len] __host__(int i) -> tcf
    {
        float t = 0.8f * ovs * static_cast<float>(i % resampled_trace_len);
        return thrust::exp(tcf(0.f, -2.f * pi * freq2 * t));
    });
    corr_downconversion_coeffs2 = hDownConv;
}

void dsp::calculateInterference(gpuvec_c& data1, gpuvec_c& data2, gpuvec_c& output, int stream_num)
{
    Npp32fc* src1 = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(data1.data()));
    const Npp32fc* coef1 = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(corr_downconversion_coeffs1.data()));
    auto status1 = nppsMul_32fc_I_Ctx(coef1, src1, static_cast<int>(data1.size()), streamContexts[stream_num]);
    check_npp_error(status1, "Error with downconversion of corr signals");

    Npp32fc* src2 = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(data2.data()));
    const Npp32fc* coef2 = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(corr_downconversion_coeffs2.data()));
    auto status2 = nppsMul_32fc_I_Ctx(coef2, src2, static_cast<int>(data2.size()), streamContexts[stream_num]);
    check_npp_error(status2, "Error with downconversion of corr signals");

    Npp32fc* dst = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(output.data()));
    auto status3 = nppsAdd_32fc_I_Ctx(src1, dst, data1.size(), streamContexts[stream_num]);
    check_npp_error(status3, "Error adding two vectors");
    auto status4 = nppsAdd_32fc_I_Ctx(src2, dst, data2.size(), streamContexts[stream_num]);
    check_npp_error(status4, "Error adding two vectors");

    // thrust::transform(
    //     data1.begin(), data1.end(), corr_downconversion_coeffs1.begin(), data1.begin(),
    //     downconv_functor());
    // thrust::transform(
    //     data2.begin(), data2.end(), corr_downconversion_coeffs2.begin(), data1.begin(),
    //     downconv_functor());
    // thrust::transform(
    //     data1.begin(), data1.end(), data2.begin(), output.begin(),
    //     thrust::plus<tcf>());    
}

void dsp::downconvert(gpuvec_c& data, int stream_num)
{
    // thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]), data.begin(), data.end(), downconversion_coeffs.begin(), data.begin(), thrust::multiplies<tcf>());
    Npp32fc* src = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(data.data()));
    const Npp32fc* coef = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(downconversion_coeffs.data()));
    auto status = nppsMul_32fc_I_Ctx(coef, src, static_cast<int>(data.size()), streamContexts[stream_num]);
    check_npp_error(status, "Error with downconversion");
}

void dsp::setDownConversionCalibrationParameters(int channel_num, float r, float phi,
    float offset_i, float offset_q)
{
    a_qi[channel_num] = std::tan(phi);
    a_qq[channel_num] = 1 / (r * std::cos(phi));
    c_i[channel_num] = offset_i;
    c_q[channel_num] = offset_q;
}

// Applies down-conversion calibration to traces
void dsp::applyDownConversionCalibration(gpuvec_c& data, cudaStream_t& stream, int channel_num)
{
    auto sync_exec_policy = thrust::cuda::par_nosync.on(stream);
    thrust::for_each(sync_exec_policy, data.begin(), data.end(), calibration_functor(a_qi[channel_num], a_qq[channel_num], c_i[channel_num], c_q[channel_num]));
}

bool dsp::hasG1() const
{
    return result_mode == ResultMode::AverageG1 || result_mode == ResultMode::AllCorrelators;
}

bool dsp::hasAllCorrelators() const
{
    return result_mode == ResultMode::AllCorrelators;
}

void dsp::requireG1(const char *getter_name) const
{
    if (!hasG1())
        throw std::runtime_error(std::string(getter_name) + " requires result_mode='average_g1' or 'all_correlators'; current result_mode='" + resultModeName(result_mode) + "'");
}

void dsp::requireAllCorrelators(const char *getter_name) const
{
    if (!hasAllCorrelators())
        throw std::runtime_error(std::string(getter_name) + " requires result_mode='all_correlators'; current result_mode='" + resultModeName(result_mode) + "'");
}

// Fills with zeros the arrays for results output in the GPU memory
void dsp::resetOutput()
{
    for (int i = 0; i < num_streams; i++)
    {
        thrust::fill(subtraction_data1[i].begin(), subtraction_data1[i].end(), tcf(0));
        thrust::fill(subtraction_data2[i].begin(), subtraction_data2[i].end(), tcf(0));
        if (hasG1())
            thrust::fill(g1[i].begin(), g1[i].end(), tcf(0));
        if (hasAllCorrelators())
        {
            thrust::fill(g1_annihilation[i].begin(), g1_annihilation[i].end(), tcf(0));
            thrust::fill(g1_creation[i].begin(), g1_creation[i].end(), tcf(0));
            thrust::fill(g1_reordered[i].begin(), g1_reordered[i].end(), tcf(0));
            thrust::fill(cross_power[i].begin(), cross_power[i].end(), tcf(0));
            thrust::fill(cross_spectrum[i].begin(), cross_spectrum[i].end(), tcf(0));
        }
        // thrust::fill(g1_filt[i].begin(), g1_filt[i].end(), tcf(0));
        // thrust::fill(g2_out[i].begin(), g2_out[i].end(), tcf(0));
        // thrust::fill(g2_out_cross_segment[i].begin(), g2_out_cross_segment[i].end(), tcf(0));
        // thrust::fill(g2_out_filtered[i].begin(), g2_out_filtered[i].end(), tcf(0));
        // thrust::fill(g2_out_filtered_cross_segment[i].begin(), g2_out_filtered_cross_segment[i].end(), tcf(0));
        // thrust::fill(data_for_correlation1[i].begin(), data_for_correlation1[i].end(), tcf(0));
        // thrust::fill(data_for_correlation2[i].begin(), data_for_correlation2[i].end(), tcf(0));
        // thrust::fill(cross_power_short[i].begin(), cross_power_short[i].end(), tcf(0));
        // thrust::fill(power1[i].begin(), power1[i].end(), tcf(0));
        // thrust::fill(power2[i].begin(), power2[i].end(), tcf(0));
        // thrust::fill(power_short[i].begin(), power_short[i].end(), tcf(0));
        // thrust::fill(interference_out[i].begin(), interference_out[i].end(), tcf(0));
        // thrust::fill(data_without_central_peak1[i].begin(), data_without_central_peak1[i].end(), tcf(0));
    }
}

int dsp::compute(const hostbuf buffer_ptr)
{
    const int stream_num = semaphore;
    switchStream();

    copyDataFromBuffer(buffer_ptr, gpu_data_buf[stream_num], stream_num);
    splitAndConvertDataToMillivolts(data1[stream_num], data2[stream_num], gpu_data_buf[stream_num], streams[stream_num]);

    // Preprocessing Data 1
    applyDownConversionCalibration(data1[stream_num], streams[stream_num], 0);
    applyFilter(data1[stream_num], firwin, stream_num, trace_length, plans[stream_num]);
    downconvert(data1[stream_num], stream_num);
    resample(data1[stream_num], data1_resampled[stream_num], streams[stream_num]);
    subtractDataFromOutput(subtraction_trace1, data1_resampled[stream_num], stream_num);
    addDataToOutput(data1_resampled[stream_num], subtraction_data1[stream_num], stream_num);
    if (hasAllCorrelators())
    {
        thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
            data1_resampled[stream_num].begin(), data1_resampled[stream_num].end(),
            data1_resampled_conj[stream_num].begin(),
            complex_conjugate());
    }

    // Preprocessing Data 2
    applyDownConversionCalibration(data2[stream_num], streams[stream_num], 1);
    applyFilter(data2[stream_num], firwin, stream_num, trace_length, plans[stream_num]);
    downconvert(data2[stream_num], stream_num);
    resample(data2[stream_num], data2_resampled[stream_num], streams[stream_num]);
    subtractDataFromOutput(subtraction_trace2, data2_resampled[stream_num], stream_num);
    addDataToOutput(data2_resampled[stream_num], subtraction_data2[stream_num], stream_num);
    if (hasG1())
    {
        thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
            data2_resampled[stream_num].begin(), data2_resampled[stream_num].end(),
            data2_resampled_conj[stream_num].begin(),
            complex_conjugate());
    }
    
    
    if (hasG1())
        calculateG1gemm(data1_resampled[stream_num], data2_resampled_conj[stream_num], g1[stream_num], cublas_handles[stream_num], op_n, op_t); // <S1* S2>
    if (hasAllCorrelators())
    {
        calculateG1gemm(data1_resampled[stream_num], data2_resampled[stream_num], g1_annihilation[stream_num], cublas_handles[stream_num], op_n, op_t); // <S1 S2>
        calculateG1gemm(data2_resampled_conj[stream_num], data1_resampled_conj[stream_num], g1_creation[stream_num], cublas_handles[stream_num], op_n, op_t); // <S1* S2*>
        calculateG1gemm(data2_resampled_conj[stream_num], data1_resampled[stream_num], g1_reordered[stream_num], cublas_handles[stream_num], op_n, op_t); // <S1* S2*>
    }
    // // Filtering left sideband
    // copyData(data1_resampled[stream_num], data_for_correlation1[stream_num], streams[stream_num]);
    // applyFilter(data_for_correlation1[stream_num], corr_firwin1, stream_num, resampled_trace_length, corr_plans[stream_num]);
    // // addDataToOutput(data_for_correlation1[stream_num], subtraction_data1[stream_num], stream_num);

    // // Filtering right sideband
    // copyData(data2_resampled[stream_num], data_for_correlation2[stream_num], streams[stream_num]);
    // applyFilter(data_for_correlation2[stream_num], corr_firwin2, stream_num, resampled_trace_length, corr_plans[stream_num]);
    // // addDataToOutput(data_for_correlation2[stream_num], subtraction_data2[stream_num], stream_num);

    if (hasAllCorrelators())
    {
        // Cross-power: conj(data1_resampled) * data2_resampled
        auto cross_power_begin = thrust::make_zip_iterator(
            data1_resampled[stream_num].begin(),
            data2_resampled[stream_num].begin(),
            cross_power[stream_num].begin());
        auto cross_power_end = thrust::make_zip_iterator(
            data1_resampled[stream_num].end(),
            data2_resampled[stream_num].end(),
            cross_power[stream_num].end());
        thrust::for_each(thrust::cuda::par_nosync.on(streams[stream_num]),
            cross_power_begin, cross_power_end,
            thrust::make_zip_function(cross_corr_accum_functor()));

        // Cross-spectrum: conj(fft(data1_resampled)) * fft(data2_resampled)
        calculateFFT(data1_resampled[stream_num], stream_num, CUFFT_FORWARD, corr_plans[stream_num]);
        calculateFFT(data2_resampled[stream_num], stream_num, CUFFT_FORWARD, corr_plans[stream_num]);
        auto cross_spectrum_begin = thrust::make_zip_iterator(
            data1_resampled[stream_num].begin(),
            data2_resampled[stream_num].begin(),
            cross_spectrum[stream_num].begin());
        auto cross_spectrum_end = thrust::make_zip_iterator(
            data1_resampled[stream_num].end(),
            data2_resampled[stream_num].end(),
            cross_spectrum[stream_num].end());
        thrust::for_each(thrust::cuda::par_nosync.on(streams[stream_num]),
            cross_spectrum_begin, cross_spectrum_end,
            thrust::make_zip_function(cross_corr_accum_functor()));
    }

    // // // Filtering out central peak
    // copyData(data1_resampled[stream_num], data_without_central_peak1[stream_num], streams[stream_num]);
    // applyFilter(data_without_central_peak1[stream_num], center_peak_win, stream_num, resampled_trace_length, corr_plans[stream_num]);
    // copyData(data2_resampled[stream_num], data_without_central_peak2[stream_num], streams[stream_num]);
    // applyFilter(data_without_central_peak2[stream_num], center_peak_win, stream_num, resampled_trace_length, corr_plans[stream_num]);

    // calculateG1gemm(data_for_correlation1[stream_num], data_for_correlation2[stream_num], g1_filt[stream_num], cublas_handles[stream_num], op_t); // <S1 S2>
    // calculateG1gemm(data_for_correlation1[stream_num], data_for_correlation2[stream_num], g1_creation[stream_num], cublas_handles[stream_num], op_c); // <S1* S2>
    // calculateG1gemm(data_without_central_peak1[stream_num], data_without_central_peak2[stream_num], g1_annihilation[stream_num], cublas_handles[stream_num], op_c); // correlation without central peak

    return stream_num;
}

void dsp::waitInputCopy(int stream_num)
{
    handleError(cudaEventSynchronize(input_copy_done[stream_num]));
}

void dsp::synchronize()
{
    for (int i = 0; i < num_streams; i++)
        handleError(cudaStreamSynchronize(streams[i]));
}

void dsp::copyData(gpuvec_c& source, gpuvec_c& dist, cudaStream_t& stream)
{
    thrust::copy(thrust::cuda::par_nosync.on(stream), source.begin(), source.end(), dist.begin());
}

// This function uploads data from the specified section of a buffer array to the GPU memory
void dsp::copyDataFromBuffer(const hostbuf buffer_ptr,
    gpubuf& dst, int stream_num)
{
    size_t width = 2 * num_channels * trace_length * sizeof(int8_t);
    size_t src_pitch = 2 * num_channels * pitch * sizeof(int8_t);
    size_t dst_pitch = width;
    size_t height = batch_size;
    handleError(cudaMemcpy2DAsync(thrust::raw_pointer_cast(dst.data()), dst_pitch,
        static_cast<const void*>(buffer_ptr), src_pitch, width, height,
        cudaMemcpyHostToDevice, streams[stream_num]));
    handleError(cudaEventRecord(input_copy_done[stream_num], streams[stream_num]));
    // cudaMemcpyAsync(thrust::raw_pointer_cast(dst.data()), reinterpret_cast<const void *>(buffer_ptr),
    //                 total_length * sizeof(char4), cudaMemcpyHostToDevice, streams[stream_num]);
}

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::splitAndConvertDataToMillivolts(gpuvec_c& data_left, gpuvec_c& data_right, const gpubuf& gpu_buf, const cudaStream_t& stream)
{
    auto begin = thrust::make_zip_iterator(gpu_buf.begin(), data_left.begin(), data_right.begin());
    auto end = thrust::make_zip_iterator(gpu_buf.end(), data_left.end(), data_right.end());
    thrust::for_each(thrust::cuda::par_nosync.on(stream),
        begin, end, thrust::make_zip_function(millivolts_functor(scale)));
}

// Applies the filter with the specified window to the data using FFT convolution
void dsp::applyFilter(gpuvec_c& data, const gpuvec_c& window, int stream_num, size_t length, cufftHandle& plan)
{
    // Step 1. Take FFT of each segment
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(data.data()));
    auto cufftstat = cufftExecC2C(plan, cufft_data, cufft_data, CUFFT_FORWARD);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 2. Multiply each segment by a window
    thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
        data.begin(), data.end(), window.begin(), data.begin(), thrust::multiplies<tcf>());
    // Step 3. Take inverse FFT of each segment
    cufftExecC2C(plan, cufft_data, cufft_data, CUFFT_INVERSE);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 4. Normalize the FFT for the output to equal the input
    thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
        data.begin(), data.end(), thrust::constant_iterator<tcf>(1.f / static_cast<float>(length)),
        data.begin(), thrust::multiplies<tcf>());
}

void dsp::calculateFFT(gpuvec_c& data, int stream_num, int direction, cufftHandle& plan)
{
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(data.data()));
    auto cufftstat = cufftExecC2C(plan, cufft_data, cufft_data, direction);
    check_cufft_error(cufftstat, "Error executing cufft");
}

// void dsp::applyFilterAlt(gpuvec_c &fftdata, const gpuvec_c &window, int stream_num, size_t length, cufftHandle &plan)
// {
//     // Step 1. Multiply each segment by a window
//     thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
//                       fftdata.begin(), fftdata.end(), window.begin(), fftdata.begin(), thrust::multiplies<tcf>());
//     // Step 3. Take inverse FFT of each segment
//     calculateFFT(fftdata, stream_num, 1, plan);
//     // Step 4. Normalize the FFT for the output to equal the input
//     thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
//                       fftdata.begin(), fftdata.end(), thrust::constant_iterator<tcf>(1.f / static_cast<float>(length)),
//                       fftdata.begin(), thrust::multiplies<tcf>());
// }

// Sums newly processed data with previous data for averaging
void dsp::addDataToOutput(const gpuvec_c& data, gpuvec_c& output, int stream_num)
{
    const Npp32fc* src = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(data.data()));
    Npp32fc* dst = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(output.data()));
    auto status = nppsAdd_32fc_I_Ctx(src, dst, data.size(), streamContexts[stream_num]);
    check_npp_error(status, "Error adding two vectors");
}

// void dsp::addDataToOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num)
// {
//     thrust::transform(thrust::cuda::par_nosync.on(streams[stream_num]), output.begin(), output.end(), data.begin(),
//         output.begin(), thrust::plus<tcf>());
// }

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(const gpuvec_c& data, gpuvec_c& output, int stream_num)
{
    const Npp32fc* src = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(data.data()));
    Npp32fc* dst = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(output.data()));
    auto status = nppsSub_32fc_I_Ctx(src, dst, data.size(), streamContexts[stream_num]);
    check_npp_error(status, "Error subtracting two vectors");
    /*thrust::transform(thrust::cuda::par_nosync.on(stream), output.begin(), output.end(), data.begin(),
        output.begin(), thrust::minus<tcf>());*/
}

void dsp::resample(const gpuvec_c& traces, gpuvec_c& resampled_traces, const cudaStream_t& stream)
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

void dsp::normalize(gpuvec_c& data, float coeff, int stream_num)
{
    Npp32fc x{ coeff, 0.f };
    nppsMulC_32fc_I_Ctx(x, reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(data.data())),
        data.size(), streamContexts[stream_num]);
}

void dsp::calculateG1(gpuvec_c& data1, gpuvec_c& data2, gpuvec_c& output, cublasHandle_t& handle)
{
    using namespace std::string_literals;

    // Compute correlation for the signal and add it to the output
    auto cublas_status = cublasCsyrkx(handle,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, resampled_trace_length, batch_size,
        &alpha, reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(data1.data())), resampled_trace_length,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(data2.data())), resampled_trace_length,
        &beta, reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status,
        "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
}

void dsp::calculateG1gemm(gpuvec_c& data1, gpuvec_c& data2, gpuvec_c& output, cublasHandle_t& handle, cublasOperation_t& op_1, cublasOperation_t& op_2)
{
    using namespace std::string_literals;
    // Compute correlation for the signal and add it to the output
    auto cublas_status = cublasCgemm3m(handle,
        op_1, op_2, resampled_trace_length, resampled_trace_length, batch_size,
        &alpha, reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(data1.data())), resampled_trace_length,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(data2.data())), resampled_trace_length,
        &beta, reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status,
        "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
}

// Calculate second-order correlation function.
// Could be used for calculating correlations between filtered signals.
// For this use dsp::makeFilterWindow and dsp::applyFilter before calling this method.
void dsp::calculateG2(gpuvec_c& data_1, gpuvec_c& data_2, gpuvec_c& cross_power, gpuvec_c& output, const cudaStream_t& stream, cublasHandle_t& handle)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream),
        data_1.begin(), data_1.end(), data_2.begin(), cross_power.begin(), cross_power_functor());
    // Calculating G2 as two-time cross power correlation
    auto cublas_status = cublasCsyrk(handle,
        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, resampled_trace_length, batch_size,
        &alpha,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(cross_power.data())), resampled_trace_length,
        &beta,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    using namespace std::string_literals;
    check_cublas_error(cublas_status,
        "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status));
}

void dsp::calculateG2gemm(gpuvec_c& data_1, gpuvec_c& data_2, gpuvec_c& cross_power, gpuvec_c& output, const cudaStream_t& stream, cublasHandle_t& handle)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream),
        data_1.begin(), data_1.end(), data_2.begin(), cross_power.begin(), cross_power_functor());
    // Calculating G2 as two-time cross power correlation
    auto cublas_status = cublasCgemm3m(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size,
        &alpha,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(cross_power.data())), resampled_trace_length,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(cross_power.data())), resampled_trace_length,
        &beta,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    using namespace std::string_literals;
    check_cublas_error(cublas_status,
        "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status));
}

void dsp::calculateG2New(gpuvec_c& data_1, gpuvec_c& data_2, gpuvec_c& cross_power, gpuvec_c& cross_power_short, gpuvec_c& output_one_segment, gpuvec_c& output_cross_segment,
    const cudaStream_t& stream, cublasHandle_t& handle)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream),
        data_1.begin(), data_1.end(), data_2.begin(), cross_power.begin(), cross_power_functor());
    thrust::copy(thrust::cuda::par_nosync.on(stream), cross_power.begin(), cross_power.end() - resampled_trace_length, cross_power_short.begin());

    auto cublas_status1 = cublasCgemm3m(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size - 1,
        &alpha,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(cross_power_short.data())), resampled_trace_length,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(cross_power.data() + resampled_trace_length)), resampled_trace_length,
        &beta,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output_cross_segment.data())), resampled_trace_length);
    // Check for errors
    using namespace std::string_literals;
    check_cublas_error(cublas_status1,
        "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status1));

    auto cublas_status2 = cublasCgemm3m(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size,
        &alpha,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(cross_power.data())), resampled_trace_length,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(cross_power.data())), resampled_trace_length,
        &beta,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output_one_segment.data())), resampled_trace_length);
    check_cublas_error(cublas_status2,
        "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status2));
}

void dsp::calculateG2Alt(gpuvec_c& data_1, gpuvec_c& data_2, gpuvec_c& power1, gpuvec_c& power2, gpuvec_c& power1_short,
    gpuvec_c& output_one_segment, gpuvec_c& output_cross_segment, const cudaStream_t& stream, cublasHandle_t& handle)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream),
        data_1.begin(), data_1.end(), data_1.begin(), power1.begin(), cross_power_functor());
    thrust::transform(thrust::cuda::par_nosync.on(stream),
        data_2.begin(), data_2.end(), data_2.begin(), power2.begin(), cross_power_functor());
    thrust::copy(thrust::cuda::par_nosync.on(stream), power1.begin(), power1.end() - resampled_trace_length, power1_short.begin());

    auto cublas_status1 = cublasCgemm3m(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size - 1,
        &alpha,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(power1_short.data())), resampled_trace_length,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(power2.data() + resampled_trace_length)), resampled_trace_length,
        &beta,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output_cross_segment.data())), resampled_trace_length);
    // Check for errors
    using namespace std::string_literals;
    check_cublas_error(cublas_status1,
        "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status1));

    auto cublas_status2 = cublasCgemm3m(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size,
        &alpha,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(power1.data())), resampled_trace_length,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(power2.data())), resampled_trace_length,
        &beta,
        reinterpret_cast<cuComplex*>(thrust::raw_pointer_cast(output_one_segment.data())), resampled_trace_length);
    check_cublas_error(cublas_status2,
        "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status2));
}

template <typename T>
thrust::host_vector<T> dsp::getCumulativeTrace(const thrust::device_vector<T>* traces, const T divisor)
{
    synchronize();
    thrust::device_vector<T> tmp(traces[0].size(), T(0));
    for (int i = 0; i < num_streams; i++)
        thrust::transform(traces[i].begin(), traces[i].end(), tmp.begin(), tmp.begin(), thrust::plus<T>());
    thrust::host_vector<T> tmp_host = tmp;
    thrust::transform(tmp_host.begin(), tmp_host.end(), tmp_host.begin(), [divisor](T x) { return x / divisor; });
    return tmp_host;
}

hostvec_c dsp::getCumulativeCorrelator(gpuvec_c g_out[4])
{
    gpuvec_c c(g_out[0].size(), tcf(0));
    synchronize();
    for (int i = 0; i < num_streams; i++)
        thrust::transform(g_out[i].begin(), g_out[i].end(), c.begin(), c.begin(), thrust::plus<tcf>());
    hostvec_c result = c;
    return result;
}

hostvec_c dsp::getG1Result()
{
    requireG1("get_g1_correlator");
    return getCumulativeTrace(g1, tcf(batch_size));
}

std::tuple<hostvec_c, hostvec_c, hostvec_c> dsp::getG1OtherResults()
{
    requireAllCorrelators("get_g1_other_correlators");
    return {
        getCumulativeTrace(g1_reordered, tcf(batch_size)),
        getCumulativeTrace(g1_creation, tcf(batch_size)),
        getCumulativeTrace(g1_annihilation, tcf(batch_size))
    };
}

__global__ void sumTracesReduce(
    const tcf* __restrict__ t_s1,
    const tcf* __restrict__ t_s2,
    const tcf* __restrict__ t_s3,
    const tcf* __restrict__ t_s4,
    tcf* __restrict__ output,
    int trace_length,
    int batch_size)
{
    int idx = blockIdx.x;
    if (idx >= trace_length) return;

    tcf local(0.0f, 0.0f);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x)
    {
        int glob_idx = i * trace_length + idx;
        local += t_s1[glob_idx] + t_s2[glob_idx] + t_s3[glob_idx] + t_s4[glob_idx];
    }

    extern __shared__ tcf smem[];
    smem[threadIdx.x] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        output[idx] = smem[0] * (1.0f / static_cast<float>(batch_size));
}

__global__ void s21Reduce(
    const tcf* __restrict__ s1_0,
    const tcf* __restrict__ s1_1,
    const tcf* __restrict__ s1_2,
    const tcf* __restrict__ s1_3,
    const tcf* __restrict__ s2_0,
    const tcf* __restrict__ s2_1,
    const tcf* __restrict__ s2_2,
    const tcf* __restrict__ s2_3,
    float2* __restrict__ out1,
    float2* __restrict__ out2,
    size_t n)
{
    float2 local1{ 0.f, 0.f };
    float2 local2{ 0.f, 0.f };

    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        tcf v1 = s1_0[i] + s1_1[i] + s1_2[i] + s1_3[i];
        local1.x += v1.real();
        local1.y += v1.imag();

        tcf v2 = s2_0[i] + s2_1[i] + s2_2[i] + s2_3[i];
        local2.x += v2.real();
        local2.y += v2.imag();
    }

    __shared__ float2 smem1[256];
    __shared__ float2 smem2[256];
    smem1[threadIdx.x] = local1;
    smem2[threadIdx.x] = local2;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            smem1[threadIdx.x].x += smem1[threadIdx.x + s].x;
            smem1[threadIdx.x].y += smem1[threadIdx.x + s].y;
            smem2[threadIdx.x].x += smem2[threadIdx.x + s].x;
            smem2[threadIdx.x].y += smem2[threadIdx.x + s].y;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&out1->x, smem1[0].x);
        atomicAdd(&out1->y, smem1[0].y);
        atomicAdd(&out2->x, smem2[0].x);
        atomicAdd(&out2->y, smem2[0].y);
    }
}

std::pair<stdvec_c, stdvec_c> dsp::getAverageField()
{
    synchronize();
    const int length = getResampledTraceLength();
    stdvec_c h_tmp1(length);
    stdvec_c h_tmp2(length);
    sumTracesReduce<<<length, 256, 256 * sizeof(tcf)>>>(
        thrust::raw_pointer_cast(subtraction_data1[0].data()),
        thrust::raw_pointer_cast(subtraction_data1[1].data()),
        thrust::raw_pointer_cast(subtraction_data1[2].data()),
        thrust::raw_pointer_cast(subtraction_data1[3].data()),
        thrust::raw_pointer_cast(tmp1.data()), length, batch_size);
    handleError(cudaGetLastError());
    sumTracesReduce<<<length, 256, 256 * sizeof(tcf)>>>(
        thrust::raw_pointer_cast(subtraction_data2[0].data()),
        thrust::raw_pointer_cast(subtraction_data2[1].data()),
        thrust::raw_pointer_cast(subtraction_data2[2].data()),
        thrust::raw_pointer_cast(subtraction_data2[3].data()),
        thrust::raw_pointer_cast(tmp2.data()), length, batch_size);
    handleError(cudaGetLastError());
    // copy from tmp1 and tmp2 to host vectors
    handleError(cudaMemcpy(h_tmp1.data(), thrust::raw_pointer_cast(tmp1.data()), length * sizeof(tcf), cudaMemcpyDeviceToHost));
    handleError(cudaMemcpy(h_tmp2.data(), thrust::raw_pointer_cast(tmp2.data()), length * sizeof(tcf), cudaMemcpyDeviceToHost));
    return { h_tmp1, h_tmp2 };
}


std::pair<std::complex<float>, std::complex<float>> dsp::getS21()
{
    synchronize();
    const size_t n = subtraction_data1[0].size();
    if (n == 0)
        return { std::complex<float>(0.f, 0.f), std::complex<float>(0.f, 0.f) };

    this->handleError(cudaMemset(s21_sum1, 0, sizeof(float2)));
    this->handleError(cudaMemset(s21_sum2, 0, sizeof(float2)));

    constexpr int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;

    s21Reduce<<<blocks, threads>>>(
        thrust::raw_pointer_cast(subtraction_data1[0].data()),
        thrust::raw_pointer_cast(subtraction_data1[1].data()),
        thrust::raw_pointer_cast(subtraction_data1[2].data()),
        thrust::raw_pointer_cast(subtraction_data1[3].data()),
        thrust::raw_pointer_cast(subtraction_data2[0].data()),
        thrust::raw_pointer_cast(subtraction_data2[1].data()),
        thrust::raw_pointer_cast(subtraction_data2[2].data()),
        thrust::raw_pointer_cast(subtraction_data2[3].data()),
        s21_sum1,
        s21_sum2,
        n);
    this->handleError(cudaGetLastError());

    float2 h1{ 0.f, 0.f };
    float2 h2{ 0.f, 0.f };
    this->handleError(cudaMemcpy(&h1, s21_sum1, sizeof(float2), cudaMemcpyDeviceToHost));
    this->handleError(cudaMemcpy(&h2, s21_sum2, sizeof(float2), cudaMemcpyDeviceToHost));

    float inv_points = 1.0f / static_cast<float>(n);
    return {
        std::complex<float>(h1.x * inv_points, h1.y * inv_points),
        std::complex<float>(h2.x * inv_points, h2.y * inv_points)
    };
}

hostvec_c dsp::getCrossPower()
{
    requireAllCorrelators("get_cross_power");
    synchronize();
    const int length = getResampledTraceLength();
    stdvec_c h_cross_power(length);
    sumTracesReduce<<<length, 256, 256 * sizeof(tcf)>>>(
        thrust::raw_pointer_cast(cross_power[0].data()),
        thrust::raw_pointer_cast(cross_power[1].data()),
        thrust::raw_pointer_cast(cross_power[2].data()),
        thrust::raw_pointer_cast(cross_power[3].data()),
        thrust::raw_pointer_cast(tmp_cross.data()),
        length,
        batch_size);
    handleError(cudaGetLastError());
    handleError(cudaMemcpy(h_cross_power.data(), thrust::raw_pointer_cast(tmp_cross.data()), length * sizeof(tcf), cudaMemcpyDeviceToHost));
    return h_cross_power;
}

hostvec_c dsp::getCrossSpectrum()
{
    requireAllCorrelators("get_cross_spectrum");
    synchronize();
    const int length = getResampledTraceLength();
    stdvec_c h_cross_spectrum(length);
    sumTracesReduce<<<length, 256, 256 * sizeof(tcf)>>>(
        thrust::raw_pointer_cast(cross_spectrum[0].data()),
        thrust::raw_pointer_cast(cross_spectrum[1].data()),
        thrust::raw_pointer_cast(cross_spectrum[2].data()),
        thrust::raw_pointer_cast(cross_spectrum[3].data()),
        thrust::raw_pointer_cast(tmp_cross.data()),
        length,
        batch_size);
    handleError(cudaGetLastError());
    handleError(cudaMemcpy(h_cross_spectrum.data(), thrust::raw_pointer_cast(tmp_cross.data()), length * sizeof(tcf), cudaMemcpyDeviceToHost));
    return h_cross_spectrum;
}

// hostvec_c dsp::getG1CrossResult()
// {
//     return getCumulativeTrace(g1_annihilation, tcf(batch_size));
// }

// hostvec_c dsp::getG1FiltResult()
// {
//     return getCumulativeTrace(g1_filt, tcf(batch_size));
// }

// hostvec_c dsp::getG1FiltConjResult()
// {
//     return getCumulativeTrace(g1_creation, tcf(batch_size));
// }

// hostvec_c dsp::getG2FullResult()
// {   
//     return getCumulativeTrace(g2_out, tcf(batch_size));
// }

// hostvec_c dsp::getG2CrossSegmentResult()
// {   
//     return getCumulativeTrace(g2_out_cross_segment, tcf(batch_size - 1));
// }

// hostvec_c dsp::getG2FilteredResult()
// {
//     return getCumulativeTrace(g2_out_filtered, tcf(batch_size));
// }

// hostvec_c dsp::getG2FilteredCrossSegmentResult()
// {   
//     return getCumulativeTrace(g2_out_filtered_cross_segment, tcf(batch_size - 1));
// }

// hostvec_c dsp::getInterferenceRsult()
// {
//     return getCumulativeTrace(interference_out, tcf(batch_size));
// }

// std::vector<hostvec_c> dsp::getCumulativeSubtrData()
// {
//     std::vector<hostvec_c> subtr_data;
//     subtr_data.push_back(getCumulativeTrace(subtraction_data1));
//     subtr_data.push_back(getCumulativeTrace(subtraction_data2));
//     return subtr_data;
// }

std::vector<hostvec_c> dsp::getCumulativeSubtrData()
{
    std::vector<hostvec_c> subtr_data;
    gpuvec_c f1(subtraction_data1[0].size(), tcf(0));
    gpuvec_c f2(subtraction_data2[0].size(), tcf(0));
    synchronize();
    for (int i = 0; i < num_streams; i++)
    {
        thrust::transform(subtraction_data1[i].begin(), subtraction_data1[i].end(), f1.begin(), f1.begin(), thrust::plus<tcf>());
        thrust::transform(subtraction_data2[i].begin(), subtraction_data2[i].end(), f2.begin(), f2.begin(), thrust::plus<tcf>());
    }

    hostvec_c s1 = f1;
    hostvec_c s2 = f2;
    subtr_data.push_back(s1);
    subtr_data.push_back(s2);
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
    scale = static_cast<float>(ampl) / 128.f; // max int8 is 127
}

void dsp::setSubtractionTrace(hostvec_c trace[num_channels])
{
    subtraction_trace1 = trace[0];
    subtraction_trace2 = trace[1];
}

void dsp::getSubtractionTrace(std::vector<stdvec_c>& trace)
{
    synchronize();
    hostvec_c h_subtr_trace1 = subtraction_trace1;
    hostvec_c h_subtr_trace2 = subtraction_trace2;
    trace.push_back(stdvec_c(h_subtr_trace1.begin(), h_subtr_trace1.end()));
    trace.push_back(stdvec_c(h_subtr_trace2.begin(), h_subtr_trace2.end()));
}

void dsp::resetSubtractionTrace()
{
    thrust::fill(subtraction_trace1.begin(), subtraction_trace1.end(), tcf(0));
    thrust::fill(subtraction_trace2.begin(), subtraction_trace2.end(), tcf(0));
}
