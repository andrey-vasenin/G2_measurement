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
#include "dsp_helpers.cuh"
#include "sidebands.cuh"

// DSP constructor
dsp::dsp(size_t len, uint64_t n, double part,
         double sample_rate, int second_oversampling) : trace_length{static_cast<size_t>(std::round((double)len * part))}, // Length of a signal or noise trace
                                                        batch_size{n},                                                     // Number of segments in a buffer (same: number of traces in data)
                                                        total_length{batch_size * trace_length},
                                                        samplerate{static_cast<float>(sample_rate)},
                                                        oversampling{second_oversampling},
                                                        resampled_trace_length{trace_length / oversampling},
                                                        resampled_total_length{total_length / oversampling},
                                                        out_size{resampled_trace_length * resampled_trace_length},
                                                        trace1_start{0},       // Start of the signal data
                                                        trace2_start{len / 2}, // Start of the noise data
                                                        pitch{len}             // Segment length in a buffer

{
    downconversion_coeffs.resize(total_length, tcf(0.f));
    corr_downconversion_coeffs1.resize(resampled_total_length, tcf(0.f));
    corr_downconversion_coeffs2.resize(resampled_total_length, tcf(0.f));
    firwin.resize(total_length, tcf(1.f)); // GPU memory for the filtering window
    average_data.resize(resampled_total_length, tcf(0.f));
    average_noise.resize(resampled_total_length, tcf(0.f));
    subtraction_data.resize(resampled_total_length, tcf(0.f));
    subtraction_noise.resize(resampled_total_length, tcf(0.f));
    // Allocate arrays on GPU for every stream
    for (int i = 0; i < num_streams; i++)
    {
        gpu_data_buf[i].resize(2 * total_length, char2{0, 0});
        gpu_noise_buf[i].resize(2 * total_length, char2{0, 0});

        // Create streams for parallel data processing
        // streams[i] = std::shared_ptr<cudaStream_t>(new cudaStream_t, streamDeleter);
        handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        check_npp_error(nppGetStreamContext(&streamContexts[i]), "Npp Error GetStreamContext");
        streamContexts[i].hStream = streams[i];

        // Allocate arrays on GPU for every channel of digitizer
        data[i].resize(total_length, tcf(0.f));
        data_resampled[i].resize(resampled_total_length, tcf(0.f));

        noise[i].resize(total_length, tcf(0.f));
        noise_resampled[i].resize(resampled_total_length, tcf(0.f));

        interference_out[i].resize(resampled_total_length, tcf(0.f));

        g1_filt[i].resize(out_size, tcf(0.f));
        g1_filt_cross_segment[i].resize(out_size, tcf(0.f));

        g1_filt_conj[i].resize(out_size, tcf(0.f));
        g1_filt_conj_cross_segment[i].resize(out_size, tcf(0.f));

        g1_without_cp[i].resize(out_size, tcf(0.f));
        g1_without_cp_cross_segment[i].resize(out_size, tcf(0.f));

        g2_filt[i].resize(out_size, tcf(0.f));
        g2_filt_cross_segment[i].resize(out_size, 0.f);

        power_short[i].resize(resampled_total_length - resampled_trace_length, 0.f);

        // plans[i] = std::shared_ptr<cufftHandle>(new cufftHandle, cufftPlanDeleter);
        // plans_resampled[i] = std::shared_ptr<cufftHandle>(new cufftHandle, cufftPlanDeleter);
        cublas_handles[i] = std::shared_ptr<cublasHandle_t>(new cublasHandle_t, cublasDeleter);
        // Initialize cuFFT plans
        check_cufft_error(cufftPlan1d(&plans[i], static_cast<int>(trace_length),
                                      CUFFT_C2C, static_cast<int>(batch_size)),
                          "Error initializing cuFFT plan\n");
        check_cufft_error(cufftPlan1d(&plans_resampled[i], static_cast<int>(resampled_trace_length),
                                      CUFFT_C2C, static_cast<int>(batch_size)),
                          "Error initializing cuFFT plan\n");
        // Assign streams to cuFFT plans
        check_cufft_error(cufftSetStream(plans[i], streams[i]),
                          "Error assigning a stream to a cuFFT plan\n");
        check_cufft_error(cufftSetStream(plans_resampled[i], streams[i]),
                          "Error assigning a stream to a cuFFT plan\n");
        // Initialize cuBLAS
        check_cublas_error(cublasCreate(cublas_handles[i].get()),
                           "Error initializing a cuBLAS handle\n");
        // Assign streams to cuBLAS handles
        check_cublas_error(cublasSetStream(*cublas_handles[i], streams[i]),
                           "Error assigning a stream to a cuBLAS handle\n");

        modules[i] = std::make_unique<sidebands_module>(resampled_trace_length, batch_size,
                                                        samplerate / static_cast<float>(second_oversampling),
                                                        plans_resampled[i], streams[i]);
    }
}

// DSP destructor
dsp::~dsp()
{
    deleteBuffer();
    for (int i = 0; i < num_streams; i++)
    {
        // Destroy cuBLAS
        // cublasDestroy(cublas_handles[i]);

        // Destroy cuFFT plans
        cufftDestroy(plans[i]);
        cufftDestroy(plans_resampled[i]);

        // Destroy GPU streams
        handleError(cudaStreamDestroy(streams[i]));
    }
}

void dsp::cufftPlanDeleter(cufftHandle *ptr)
{
    if (ptr)
    {
        cufftDestroy(*ptr);
        delete ptr;
    }
}

void dsp::cublasDeleter(cublasHandle_t *ptr)
{
    if (ptr)
    {
        cublasDestroy(*ptr);
        delete ptr;
    }
}

void dsp::streamDeleter(cudaStream_t *ptr)
{
    if (ptr)
    {
        handleError(cudaStreamDestroy(*ptr));
        delete ptr;
    }
}

// Set filtering window for digital processing
void dsp::setFirwin(float cutoff_l, float cutoff_r)
{
    makeFilterWindow(cutoff_l, cutoff_r, firwin, trace_length, total_length, samplerate);
}

void dsp::setFirwin(hostvec_c window)
{
    firwin = window;
}

// Creates a rectangular window with specified cutoff frequencies for the further usage in a filter
// Frequencies in MHz
void dsp::makeFilterWindow(float cutoff_l, float cutoff_r, gpuvec_c &window, size_t trace_len, size_t total_len, float samplerate)
{
    using namespace std::complex_literals;
    hostvec_c hFirwin(total_len);
    float fs = samplerate * 1e-6;
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
    this->handleError(cudaMallocHost((void **)&buffer, size));
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
    hostvec_c hDownConv(total_length);
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
                     [=] __host__(int i) -> tcf
                     {
                         float t = 0.8f * ovs * static_cast<float>(i % trace_length);
                         return thrust::exp(tcf(0.f, -2.f * pi * frequency * t));
                     });
    downconversion_coeffs = hDownConv;
}

void dsp::setCorrDowncovertCoeffs(float freq1, float freq2, int oversampling)
{
    const float pi = std::acos(-1.f);
    float ovs = static_cast<float>(oversampling);
    hostvec_c hDownConv(resampled_total_length);
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
                     [=] __host__(int i) -> tcf
                     {
                         float t = 0.8f * ovs * static_cast<float>(i % resampled_trace_length);
                         return thrust::exp(tcf(0.f, -2.f * pi * freq1 * t));
                     });
    corr_downconversion_coeffs1 = hDownConv;

    thrust::fill(hDownConv.begin(), hDownConv.end(), tcf(0.f));
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
                     [=] __host__(int i) -> tcf
                     {
                         float t = 0.8f * ovs * static_cast<float>(i % resampled_trace_length);
                         return thrust::exp(tcf(0.f, -2.f * pi * freq2 * t));
                     });
    corr_downconversion_coeffs2 = hDownConv;
}

void dsp::downconvert(gpuvec_c &data, int stream_num)
{
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
    thrust::fill(average_data.begin(), average_data.end(), tcf(0));
    thrust::fill(average_noise.begin(), average_noise.end(), tcf(0));
    for (int i = 0; i < num_streams; i++)
    {

        thrust::fill(data_resampled[i].begin(), data_resampled[i].end(), tcf(0));
        thrust::fill(noise_resampled[i].begin(), noise_resampled[i].end(), tcf(0));

        thrust::fill(g1_filt[i].begin(), g1_filt[i].end(), tcf(0));
        thrust::fill(g1_filt_cross_segment[i].begin(), g1_filt_cross_segment[i].end(), tcf(0));

        thrust::fill(g1_filt_conj[i].begin(), g1_filt_conj[i].end(), tcf(0));
        thrust::fill(g1_filt_conj_cross_segment[i].begin(), g1_filt_conj_cross_segment[i].end(), tcf(0));

        thrust::fill(g2_filt[i].begin(), g2_filt[i].end(), tcf(0));
        thrust::fill(g2_filt_cross_segment[i].begin(), g2_filt_cross_segment[i].end(), 0);

        thrust::fill(g1_without_cp[i].begin(), g1_without_cp[i].end(), tcf(0));
        thrust::fill(g1_without_cp_cross_segment[i].begin(), g1_without_cp_cross_segment[i].end(), tcf(0));
        modules[i]->reset();
        thrust::fill(power_short[i].begin(), power_short[i].end(), 0);

        thrust::fill(interference_out[i].begin(), interference_out[i].end(), tcf(0));
    }
}

void dsp::compute(const hostbuf buffer_ptr)
{
    const int stream_num = semaphore;
    switchStream();

    loadDataToGPUwithPitchAndOffset(buffer_ptr, gpu_data_buf[stream_num], pitch, trace1_start, stream_num);
    loadDataToGPUwithPitchAndOffset(buffer_ptr, gpu_noise_buf[stream_num], pitch, trace2_start, stream_num);

    convertDataToMillivolts(data[stream_num], gpu_data_buf[stream_num], streams[stream_num]);
    convertDataToMillivolts(noise[stream_num], gpu_noise_buf[stream_num], streams[stream_num]);

    applyDownConversionCalibration(data[stream_num], streams[stream_num]);
    applyDownConversionCalibration(noise[stream_num], streams[stream_num]);

    applyFilter(data[stream_num], firwin, trace_length, streams[stream_num], plans[stream_num]);
    applyFilter(noise[stream_num], firwin, trace_length, streams[stream_num], plans[stream_num]);

    downconvert(data[stream_num], stream_num);
    downconvert(noise[stream_num], stream_num);

    addDataToOutput(data[stream_num], average_data, stream_num);
    addDataToOutput(noise[stream_num], average_noise, stream_num);

    resample(data[stream_num], data_resampled[stream_num], streams[stream_num]);
    resample(noise[stream_num], noise_resampled[stream_num], streams[stream_num]);

    modules[stream_num]->compute(data_resampled[stream_num], noise_resampled[stream_num]);
}

// This function uploads data from the specified section of a buffer array to the GPU memory
void dsp::loadDataToGPUwithPitchAndOffset(const hostbuf buffer_ptr,
                                          gpubuf &gpu_buf, size_t pitch, size_t offset, int stream_num)
{
    size_t width = 2 * size_t(trace_length) * sizeof(int8_t);
    size_t src_pitch = 2 * pitch * sizeof(int8_t);
    size_t dst_pitch = width;
    size_t shift = 2 * offset;
    handleError(cudaMemcpy2DAsync(thrust::raw_pointer_cast(gpu_buf.data()), dst_pitch,
                                  static_cast<const void *>(buffer_ptr + shift), src_pitch, width, batch_size,
                                  cudaMemcpyHostToDevice, streams[stream_num]));
}

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMillivolts(gpuvec_c &data, const gpubuf &gpu_buf, const cudaStream_t &stream)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      gpu_buf.begin(), gpu_buf.end(), data.begin(), millivolts_functor(scale));
}

// Applies the filter with the specified window to the data using FFT convolution
void dsp::applyFilter(gpuvec_c &data, gpuvec_c &window, size_t size, cudaStream_t &stream, cufftHandle &plan)
{
    // Step 1. Take FFT of each segment
    cufftComplex *cufft_data = reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(data.data()));
    auto cufftstat = cufftExecC2C(plan, cufft_data, cufft_data, CUFFT_FORWARD);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 2. Multiply each segment by a window
    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      data.begin(), data.end(), window.begin(), data.begin(), thrust::multiplies<tcf>());
    // Step 3. Take inverse FFT of each segment
    cufftExecC2C(plan, cufft_data, cufft_data, CUFFT_INVERSE);
    check_cufft_error(cufftstat, "Error executing cufft");
    // Step 4. Normalize the FFT for the output to equal the input
    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      data.begin(), data.end(), thrust::constant_iterator<tcf>(1.f / static_cast<float>(size)),
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

void dsp::calculatePower(gpuvec_c &data, gpuvec_c &noise, gpuvec &output, cudaStream_t &stream)
{
    thrust::for_each(thrust::cuda::par_nosync.on(stream),
                     thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
                     thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
                     thrust::make_zip_function(power_functor()));
}

void dsp::calculateInterference(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &noise1, gpuvec_c &noise2, gpuvec_c &output, int stream_num)
{
    Npp32fc *src1 = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(data1.data()));
    const Npp32fc *coef1 = reinterpret_cast<const Npp32fc *>(thrust::raw_pointer_cast(corr_downconversion_coeffs1.data()));
    auto status1 = nppsMul_32fc_I_Ctx(coef1, src1, static_cast<int>(data1.size()), streamContexts[stream_num]);
    check_npp_error(status1, "Error with downconversion of corr signals");

    Npp32fc *src2 = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(data2.data()));
    const Npp32fc *coef2 = reinterpret_cast<const Npp32fc *>(thrust::raw_pointer_cast(corr_downconversion_coeffs2.data()));
    auto status2 = nppsMul_32fc_I_Ctx(coef2, src2, static_cast<int>(data2.size()), streamContexts[stream_num]);
    check_npp_error(status2, "Error with downconversion of corr signals");

    Npp32fc *nrc1 = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(noise1.data()));
    const Npp32fc *coef3 = reinterpret_cast<const Npp32fc *>(thrust::raw_pointer_cast(corr_downconversion_coeffs1.data()));
    auto status3 = nppsMul_32fc_I_Ctx(coef3, nrc1, static_cast<int>(noise1.size()), streamContexts[stream_num]);
    check_npp_error(status3, "Error with downconversion of corr signals");

    Npp32fc *nrc2 = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(noise2.data()));
    const Npp32fc *coef4 = reinterpret_cast<const Npp32fc *>(thrust::raw_pointer_cast(corr_downconversion_coeffs2.data()));
    auto status4 = nppsMul_32fc_I_Ctx(coef4, nrc2, static_cast<int>(noise2.size()), streamContexts[stream_num]);
    check_npp_error(status4, "Error with downconversion of corr signals");

    Npp32fc *dst = reinterpret_cast<Npp32fc *>(thrust::raw_pointer_cast(output.data()));
    auto status5 = nppsAdd_32fc_I_Ctx(src1, dst, data1.size(), streamContexts[stream_num]);
    check_npp_error(status5, "Error adding two vectors");
    auto status6 = nppsAdd_32fc_I_Ctx(src2, dst, data2.size(), streamContexts[stream_num]);
    check_npp_error(status4, "Error adding two vectors");

    auto status7 = nppsSub_32fc_I_Ctx(nrc1, dst, noise1.size(), streamContexts[stream_num]);
    check_npp_error(status7, "Error subtracting two vectors");
    auto status8 = nppsSub_32fc_I_Ctx(nrc2, dst, noise2.size(), streamContexts[stream_num]);
    check_npp_error(status8, "Error subtracting two vectors");
}

void dsp::calculateG1(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &noise1, gpuvec_c &noise2, gpuvec_c &output, cublasHandle_t &handle, cublasOperation_t &op)
{
    using namespace std::string_literals;
    // Compute correlation for the signal and add it to the output
    auto cublas_status1 = cublasCgemm3m(handle,
                                        CUBLAS_OP_N, op, resampled_trace_length, resampled_trace_length, batch_size,
                                        &alpha_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(data1.data())), resampled_trace_length,
                                        reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(data2.data())), resampled_trace_length,
                                        &beta_data_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status1,
                       "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status1));

    auto cublas_status2 = cublasCgemm3m(handle,
                                        CUBLAS_OP_N, op, resampled_trace_length, resampled_trace_length, batch_size,
                                        &alpha_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(noise1.data())), resampled_trace_length,
                                        reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(noise2.data())), resampled_trace_length,
                                        &beta_noise_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status2,
                       "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status2));
}

void dsp::calculateG1cs(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &noise1, gpuvec_c &noise2,
                        gpuvec_c &data1_short, gpuvec_c &noise1_short, gpuvec_c &output, gpuvec_c &output_cs, cublasHandle_t &handle, cublasOperation_t &op, const cudaStream_t &stream)
{
    using namespace std::string_literals;
    // Compute correlation for the signal and add it to the output
    auto cublas_status1 = cublasCgemm3m(handle,
                                        CUBLAS_OP_N, op, resampled_trace_length, resampled_trace_length, batch_size,
                                        &alpha_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(data1.data())), resampled_trace_length,
                                        reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(data2.data())), resampled_trace_length,
                                        &beta_data_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status1,
                       "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status1));

    auto cublas_status2 = cublasCgemm3m(handle,
                                        CUBLAS_OP_N, op, resampled_trace_length, resampled_trace_length, batch_size,
                                        &alpha_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(noise1.data())), resampled_trace_length,
                                        reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(noise2.data())), resampled_trace_length,
                                        &beta_noise_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status2,
                       "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status2));

    thrust::copy(thrust::cuda::par_nosync.on(stream), data1.begin(), data1.end() - resampled_trace_length, data1_short.begin());
    auto cublas_status3 = cublasCgemm3m(handle,
                                        CUBLAS_OP_N, op, resampled_trace_length, resampled_trace_length, batch_size - 1,
                                        &alpha_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(data1_short.data())), resampled_trace_length,
                                        reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(data2.data() + resampled_trace_length)), resampled_trace_length,
                                        &beta_data_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output_cs.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status3,
                       "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status3));

    thrust::copy(thrust::cuda::par_nosync.on(stream), noise1.begin(), noise1.end() - resampled_trace_length, noise1_short.begin());
    auto cublas_status4 = cublasCgemm3m(handle,
                                        CUBLAS_OP_N, op, resampled_trace_length, resampled_trace_length, batch_size - 1,
                                        &alpha_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(noise1_short.data())), resampled_trace_length,
                                        reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(noise2.data() + resampled_trace_length)), resampled_trace_length,
                                        &beta_noise_c, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output_cs.data())), resampled_trace_length);
    // Check for errors
    check_cublas_error(cublas_status4,
                       "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status4));
}

void dsp::calculateG2(gpuvec &power1, gpuvec &power2, gpuvec &output, cublasHandle_t &handle)
{
    using namespace std::string_literals;

    auto cublas_status = cublasSgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size,
                                     &alpha_f,
                                     thrust::raw_pointer_cast(power1.data()), resampled_trace_length,
                                     thrust::raw_pointer_cast(power2.data()), resampled_trace_length,
                                     &beta_f,
                                     thrust::raw_pointer_cast(output.data()), resampled_trace_length);
    check_cublas_error(cublas_status,
                       "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status));
}

// Calculate second-order correlation function.
void dsp::calculateG2csAlt(gpuvec &power1, gpuvec &power2, gpuvec &power1_short,
                           gpuvec &output_one_segment, gpuvec &output_cross_segment, const cudaStream_t &stream, cublasHandle_t &handle)
{
    thrust::copy(thrust::cuda::par_nosync.on(stream), power1.begin(), power1.end() - resampled_trace_length, power1_short.begin());
    auto cublas_status1 = cublasSgemm(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size - 1,
                                      &alpha_f,
                                      thrust::raw_pointer_cast(power1_short.data()), resampled_trace_length,
                                      thrust::raw_pointer_cast(power2.data() + resampled_trace_length), resampled_trace_length,
                                      &beta_f,
                                      thrust::raw_pointer_cast(output_cross_segment.data()), resampled_trace_length);
    // Check for errors
    using namespace std::string_literals;
    check_cublas_error(cublas_status1,
                       "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status1));

    auto cublas_status2 = cublasSgemm(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_T, resampled_trace_length, resampled_trace_length, batch_size,
                                      &alpha_f,
                                      thrust::raw_pointer_cast(power1.data()), resampled_trace_length,
                                      thrust::raw_pointer_cast(power2.data()), resampled_trace_length,
                                      &beta_f,
                                      thrust::raw_pointer_cast(output_one_segment.data()), resampled_trace_length);
    check_cublas_error(cublas_status2,
                       "Error of rank-2 update (data) with code #"s + std::to_string(cublas_status2));
}

template <typename T>
thrust::host_vector<T> dsp::getCumulativeTrace(const thrust::device_vector<T> *traces, size_t batch_size)
{
    handleError(cudaDeviceSynchronize());
    auto tmp = sumOverStreams(traces);
    return sumOverBatch(tmp, batch_size);
}

template <typename T>
thrust::device_vector<T> dsp::sumOverStreams(const thrust::device_vector<T> *traces)
{
    thrust::device_vector<T> tmp(traces->size(), T(0.f));
    for (int i = 0; i < num_streams; i++)
        thrust::transform(traces[i].begin(), traces[i].end(), tmp.begin(), tmp.begin(), thrust::plus<T>());
    divideBy(tmp, T(num_streams));
    return tmp;
}

template <typename T>
thrust::host_vector<T> dsp::sumOverBatch(const thrust::device_vector<T> &trace, size_t batch_size)
{
    size_t N = trace.size() / batch_size;
    thrust::host_vector<T> host_trace(N);
    using iter = typename thrust::device_vector<T>::const_iterator;
    for (size_t j = 0; j < N; ++j)
    {
        strided_range<iter> tmp_iter(trace.begin() + j, trace.end(), N);
        host_trace[j] = thrust::reduce(tmp_iter.begin(), tmp_iter.end(), T(0.f), thrust::plus<T>());
    }
    divideBy(host_trace, T(batch_size));
    return host_trace;
}

template <typename VectorT, typename T>
void dsp::divideBy(VectorT &trace, T div)
{
    static_assert(std::is_same<typename VectorT::value_type, T>::value,
                  "Vector element type must match T");

    thrust::transform(trace.begin(), trace.end(),
                      thrust::make_constant_iterator(div),
                      trace.begin(),
                      thrust::divides<T>());
}

std::pair<hostvec_c, hostvec_c> dsp::getG1FiltResult()
{
    hostvec_c h_g1_filt_cross_segment = sumOverStreams(g1_filt_cross_segment);
    hostvec_c h_g1_filt = sumOverStreams(g1_filt);
    return std::make_pair(h_g1_filt, h_g1_filt_cross_segment);
}

std::pair<hostvec_c, hostvec_c> dsp::getG1FiltConjResult()
{
    hostvec_c h_filt_conj = sumOverStreams(g1_filt_conj);
    hostvec_c h_g1_filt_conj_cross_segment = sumOverStreams(g1_filt_conj_cross_segment);
    return std::make_pair(h_filt_conj, h_g1_filt_conj_cross_segment);
}

std::pair<hostvec_c, hostvec_c> dsp::getG1WithoutCPResult()
{
    hostvec_c h_g1_without_cp = sumOverStreams(g1_without_cp);
    hostvec_c h_g1_without_cp_cross_segment = sumOverStreams(g1_without_cp_cross_segment);
    return std::make_pair(h_g1_without_cp, h_g1_without_cp_cross_segment);
}

std::pair<hostvec_c, hostvec> dsp::getG2FilteredResult()
{
    hostvec_c h_g2_filt = sumOverStreams(g2_filt);
    hostvec h_g2_filt_cross_segment = sumOverStreams(g2_filt_cross_segment);
    return std::make_pair(h_g2_filt, h_g2_filt_cross_segment);
}

std::vector<hostvec> dsp::getRealResults()
{
    std::vector<hostvec> results;
    for (int i = 0; i < num_streams; i++)
    {
        auto stream_results = modules[i]->getRealResults();
        if (stream_results.empty())
            throw std::runtime_error("No real-valued results returned");
        if (i == 0)
            results.insert(results.end(), stream_results.begin(), stream_results.end());
        else
            for (int j = 0; j < stream_results.size(); j++)
                thrust::transform(stream_results[j].begin(),
                                  stream_results[j].end(),
                                  results[j].begin(),
                                  results[j].begin(), thrust::plus<float>());
    }
    for (int j = 0; j < results.size(); j++)
        divideBy(results[j], static_cast<float>(num_streams));
    return results;
}

std::vector<hostvec_c> dsp::getComplexResults()
{
    std::vector<hostvec_c> results;
    for (int i = 0; i < num_streams; i++)
    {
        auto stream_results = modules[i]->getComplexResults();
        if (stream_results.empty())
            throw std::runtime_error("No complex-valued results returned");
        if (i == 0)
            results.insert(results.end(), stream_results.begin(), stream_results.end());
        else
            for (int j = 0; j < stream_results.size(); j++)
                thrust::transform(stream_results[j].begin(),
                                  stream_results[j].end(),
                                  results[j].begin(),
                                  results[j].begin(), thrust::plus<tcf>());
    }
    for (int j = 0; j < results.size(); j++)
        divideBy(results[j], tcf(static_cast<float>(num_streams)));
    return results;
}

hostvec_c dsp::getInterferenceResult()
{
    return getCumulativeTrace(interference_out, batch_size);
}

std::pair<hostvec_c, hostvec_c> dsp::getAverageData()
{
    hostvec_c h_ad = average_data;
    hostvec_c h_an = average_noise;
    return {h_ad, h_an};
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
    subtraction_data = trace[0];
    subtraction_noise = trace[1];
}

void dsp::getSubtractionTrace(std::vector<stdvec_c> &trace)
{
    hostvec_c h_subtr_trace1 = subtraction_data;
    hostvec_c h_subtr_trace2 = subtraction_noise;
    trace.push_back(stdvec_c(h_subtr_trace1.begin(), h_subtr_trace1.end()));
    trace.push_back(stdvec_c(h_subtr_trace2.begin(), h_subtr_trace2.end()));
}

void dsp::resetSubtractionTrace()
{
    thrust::fill(average_data.begin(), average_data.end(), tcf(0));
    thrust::fill(average_noise.begin(), average_noise.end(), tcf(0));
}