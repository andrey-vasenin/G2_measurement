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


inline void check_cufft_error(cufftResult cufft_err, std::string &&msg)
{
#ifdef NDEBUG

    if (cufft_err != CUFFT_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_cublas_error(cublasStatus_t err, std::string &&msg)
{
#ifdef NDEBUG

    if (err != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(msg);

#endif // NDEBUG
}

inline void check_npp_error(NppStatus err, std::string &&msg)
{
#ifdef NDEBUG
    if (err != NPP_SUCCESS)
        throw std::runtime_error(msg);
#endif // NDEBUG
}

template <typename T> 
inline void print_vector(thrust::device_vector<T> & vec, int n) 
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
dsp::dsp(size_t len, uint64_t n, double part, int K_,
    double samplerate, int second_oversampling) : 
    trace_length{static_cast<size_t>(std::round((double)len * part))}, // Length of a signal or noise trace
    batch_size{n},                                    // Number of segments in a buffer (same: number of traces in data)
    total_length{batch_size * trace_length},
    oversampling{ second_oversampling },
    resampled_trace_length{ trace_length / oversampling },
    resampled_total_length{ total_length / oversampling },
    out_size{trace_length * trace_length},
    trace1_start{0},       // Start of the signal data
    trace2_start{len / 2}, // Start of the noise data
    pitch{len}           // Segment length in a buffer
                                       
{
    //firwin.resize(total_length); // GPU memory for the filtering window
    subtraction_trace.resize(total_length);
    subtraction_offs.resize(total_length);
    thrust::fill(subtraction_trace.begin(), subtraction_trace.end(), tcf(0.f));
    downconversion_coeffs.resize(total_length);

    // Setup multitaper
    K = K_;
    tapers.resize(K);

    // Streams
    for (int i = 0; i < num_streams; i++)
    {
        // Create streams for parallel data processing
        handleError(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        check_npp_error(nppGetStreamContext(&streamContexts[i]), "Npp Error GetStreamContext");
        streamContexts[i].hStream = streams[i];

        // Allocate arrays on GPU for every stream
        gpu_data_buf[i].resize(2 * total_length);
        gpu_noise_buf[i].resize(2 * total_length);
        data[i].resize(total_length);
        data_resampled[i].resize(resampled_total_length);
        subtraction_data[i].resize(total_length);
        noise[i].resize(total_length);
        noise_resampled[i].resize(resampled_total_length);
        subtraction_noise[i].resize(total_length);
        power[i].resize(total_length);
        field[i].resize(total_length);
        //out[i].resize(out_size);
        taperedData[i].resize(resampled_total_length);
        taperedNoise[i].resize(resampled_total_length);
        data_fft[i].resize(resampled_total_length);
        noise_fft[i].resize(resampled_total_length);
        spectrum[i].resize(resampled_total_length);
        periodogram[i].resize(total_length);

        // Initialize cuFFT plans
        check_cufft_error(cufftPlan1d(&plans[i], trace_length, CUFFT_C2C, batch_size),
                          "Error initializing cuFFT plan\n");
        check_cufft_error(cufftPlan1d(&multitaper_plans[i], resampled_trace_length, CUFFT_C2C, batch_size),
            "Error initializing cuFFT plan\n");

        // Assign streams to cuFFT plans
        check_cufft_error(cufftSetStream(plans[i], streams[i]),
                          "Error assigning a stream to a cuFFT plan\n");
        check_cufft_error(cufftSetStream(multitaper_plans[i], streams[i]),
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
    resetOutput();
    resetSubtractionTrace();
}

// DSP destructor
dsp::~dsp()
{
    deleteBuffer();
    for (int i = 0; i < num_streams; i++)
    {
        // Destroy cuBLAS
        cublasDestroy(cublas_handles[i]);
        cublasDestroy(cublas_handles2[i]);

        // Destroy cuFFT plans
        cufftDestroy(plans[i]);
        cufftDestroy(multitaper_plans[i]);

        // Destroy GPU streams
        handleError(cudaStreamDestroy(streams[i]));
    }
}

// Creates a rectangular window with specified cutoff frequencies for the further usage in a filter
void dsp::setFirwin(float cutoff_l, float cutoff_r, int oversampling)
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
    firwin = hFirwin;
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

void dsp::deleteBuffer() {
    this->handleError(cudaFreeHost(buffer));
};

void dsp::setIntermediateFrequency(float frequency, int oversampling)
{
    const float pi = std::acos(-1.f);
    float ovs = static_cast<float>(oversampling);
    hostvec_c hDownConv(total_length);
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
        [=] __host__ (int i) -> tcf {
            float t = 0.8 * ovs * static_cast<float>(i % trace_length);
            return thrust::exp(tcf(0, -2 * pi * frequency * t));
        });
    downconversion_coeffs = hDownConv;
}

void dsp::downconvert(gpuvec_c &data, int stream_num)
{
    // thrust::transform(thrust::cuda::par_nosync.on(stream), data.begin(), data.end(), downconversion_coeffs.begin(), data.begin(), downconv_functor());
    Npp32fc* src = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(data.data()));
    const Npp32fc* coef = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(downconversion_coeffs.data()));
    auto status = nppsMul_32fc_I_Ctx(coef, src, data.size(), streamContexts[stream_num]);
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
void dsp::applyDownConversionCalibration(gpuvec_c& data, cudaStream_t &stream)
{
    auto sync_exec_policy = thrust::cuda::par_nosync.on(stream);
    thrust::for_each(sync_exec_policy, data.begin(), data.end(), calibration_functor(a_qi, a_qq, c_i, c_q));
}

hostbuf dsp::getBuffer()
{
    return buffer;
}

// Fills with zeros the arrays for cumulative field and power in the GPU memory
void dsp::resetOutput()
{
    for (int i = 0; i < num_streams; i++)
    {
        //thrust::fill(out[i].begin(), out[i].end(), tcf(0));
        thrust::fill(field[i].begin(), field[i].end(), tcf(0));
        thrust::fill(power[i].begin(), power[i].end(), 0.f);
        thrust::fill(spectrum[i].begin(), spectrum[i].end(), 0.f);
        thrust::fill(periodogram[i].begin(), periodogram[i].end(), 0.f);
        thrust::fill(data_fft[i].begin(), data_fft[i].end(), tcf(0));
        thrust::fill(noise_fft[i].begin(), noise_fft[i].end(), tcf(0));
        thrust::fill(subtraction_data[i].begin(), subtraction_data[i].end(), tcf(0));
        thrust::fill(subtraction_noise[i].begin(), subtraction_noise[i].end(), tcf(0));
    }
}

void dsp::compute(const hostbuf buffer_ptr)
{
    const int stream_num = semaphore;
    switchStream();
    loadDataToGPUwithPitchAndOffset(buffer_ptr, gpu_data_buf[stream_num], pitch, trace1_start, stream_num);
    loadDataToGPUwithPitchAndOffset(buffer_ptr, gpu_noise_buf[stream_num], pitch, trace2_start, stream_num);
    convertDataToMillivolts(data[stream_num], gpu_data_buf[stream_num], streams[stream_num]); // error is here
    convertDataToMillivolts(noise[stream_num], gpu_noise_buf[stream_num], streams[stream_num]);
    applyDownConversionCalibration(data[stream_num], streams[stream_num]);
    applyDownConversionCalibration(noise[stream_num], streams[stream_num]);
    applyFilter(data[stream_num], firwin, stream_num);
    applyFilter(noise[stream_num], firwin, stream_num);
    downconvert(data[stream_num], stream_num);
    downconvert(noise[stream_num], stream_num);

    subtractDataFromOutput(subtraction_trace, data[stream_num], stream_num);
    subtractDataFromOutput(subtraction_offs, noise[stream_num], stream_num);

    addDataToOutput(data[stream_num], subtraction_data[stream_num], stream_num);
    addDataToOutput(noise[stream_num], subtraction_noise[stream_num], stream_num);

    calculateField(data[stream_num], noise[stream_num],
        field[stream_num], streams[stream_num]);
    calculatePower(data[stream_num], noise[stream_num], power[stream_num], streams[stream_num]);
    //calculateG1(data_calibrated[stream_num], noise_calibrated[stream_num],
    //    out[stream_num], cublas_handles[stream_num]);
    resample(data[stream_num], data_resampled[stream_num], streams[stream_num]);
    resample(noise[stream_num], noise_resampled[stream_num], streams[stream_num]);
    calculateMultitaperSpectrum(data_resampled[stream_num], noise_resampled[stream_num],
        data_fft[stream_num], noise_fft[stream_num], spectrum[stream_num], stream_num);
    calculatePeriodogram(data[stream_num], noise[stream_num],
        periodogram[stream_num], stream_num);
}

// This function uploads data from the specified section of a buffer array to the GPU memory
void dsp::loadDataToGPUwithPitchAndOffset(const hostbuf buffer_ptr,
    gpubuf& gpu_buf, size_t pitch, size_t offset, int stream_num)
{
    size_t width = 2 * size_t(trace_length) * sizeof(int8_t);
    size_t src_pitch = 2 * pitch * sizeof(int8_t);
    size_t dst_pitch = width;
    size_t shift = 2 * offset;
    handleError(cudaMemcpy2DAsync(thrust::raw_pointer_cast(gpu_buf.data()), dst_pitch,
                                  static_cast<const void*>(buffer_ptr + shift), src_pitch, width, batch_size,
                                  cudaMemcpyHostToDevice, streams[stream_num]));
}

// Converts bytes into 32-bit floats with mV dimensionality
void dsp::convertDataToMillivolts(gpuvec_c& data, const gpubuf& gpu_buf, const cudaStream_t &stream)
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
    const Npp32fc* src = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(data.data()));
    Npp32fc* dst = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(output.data()));
    auto status = nppsAdd_32fc_I_Ctx(src, dst, data.size(), streamContexts[stream_num]);
    check_npp_error(status, "Error adding two vectors");
}

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

// Calculates the field from the data in the GPU memory
void dsp::calculateField(const gpuvec_c& data, const gpuvec_c& noise, gpuvec_c& output, const cudaStream_t &stream)
{
    thrust::for_each(thrust::cuda::par_nosync.on(stream),
        thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
        thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
        thrust::make_zip_function(field_functor()));
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

// Calculates the power from the data in the GPU memory
void dsp::calculatePower(const gpuvec_c& data, const gpuvec_c& noise, gpuvec& output, const cudaStream_t& stream)
{
    thrust::for_each(thrust::cuda::par_nosync.on(stream),
        thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
        thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
        thrust::make_zip_function(power_functor()));
}

void dsp::calculateG1(gpuvec_c& data, gpuvec_c& noise, gpuvec_c& output, cublasHandle_t &handle)
{
    using namespace std::string_literals;

    const float alpha_data = 1;   // this alpha multiplies the result to be added to the output
    const float alpha_noise = -1; // this alpha multiplies the result to be added to the output
    const float beta = 1;
    // Compute correlation for the signal and add it to the output
    auto cublas_status = cublasCherk(handle,
                                     CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                     &alpha_data, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(data.data())), trace_length,
                                     &beta, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output.data())), trace_length);
    // Check for errors
    check_cublas_error(cublas_status,
        "Error of rank-1 update (data) with code #"s + std::to_string(cublas_status));
    // Compute correlation for the noise and subtract it from the output
    cublas_status = cublasCherk(handle,
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, trace_length, batch_size,
                                &alpha_noise, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(noise.data())), trace_length,
                                &beta, reinterpret_cast<cuComplex *>(thrust::raw_pointer_cast(output.data())), trace_length);
    // Check for errors
    check_cublas_error(cublas_status,
        "Error of rank-1 update (noise) with code #"s + std::to_string(cublas_status));
}

void dsp::calculatePeriodogram(gpuvec_c& data, gpuvec_c& noise, gpuvec& output, int stream_num)
{
    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(data.data()));
    auto cufftstat1 = cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_FORWARD);
    check_cufft_error(cufftstat1, "Error executing cufft");

    cufftComplex* cufft_noise = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(noise.data()));
    auto cufftstat2 = cufftExecC2C(plans[stream_num], cufft_noise, cufft_noise, CUFFT_FORWARD);
    check_cufft_error(cufftstat2, "Error executing cufft");

    thrust::for_each(thrust::cuda::par_nosync.on(streams[stream_num]),
        thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
        thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
        thrust::make_zip_function(power_functor()));
}

void dsp::calculateMultitaperSpectrum(const gpuvec_c& data, const gpuvec_c& noise, gpuvec_c& signal_field_spectra,
    gpuvec_c& noise_field_spectra, gpuvec& power_spectra, int stream_num)
{
    for (size_t i = 0; i < K; ++i) {
        // 1. Windowing the Signal with Tapers
        const Npp32f* src1_t = reinterpret_cast<const Npp32f*>(thrust::raw_pointer_cast(tapers[i].data()));
        const Npp32fc* src2_d = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(data.data()));
        Npp32fc* dst_d = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(taperedData[stream_num].data()));
        nppsMul_32f32fc_Ctx(src1_t, src2_d, dst_d, data.size(), streamContexts[stream_num]);
        const Npp32fc* src2_n = reinterpret_cast<const Npp32fc*>(thrust::raw_pointer_cast(noise.data()));
        Npp32fc* dst_n = reinterpret_cast<Npp32fc*>(thrust::raw_pointer_cast(taperedNoise[stream_num].data()));
        nppsMul_32f32fc_Ctx(src1_t, src2_n, dst_n, data.size(), streamContexts[stream_num]);
        // 2. FFT
        auto cufft_tapered_data = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(taperedData[stream_num].data()));
        auto cufft_tapered_noise = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(taperedNoise[stream_num].data()));
        cufftExecC2C(multitaper_plans[stream_num], cufft_tapered_data, cufft_tapered_data, CUFFT_FORWARD);
        cufftExecC2C(multitaper_plans[stream_num], cufft_tapered_noise, cufft_tapered_noise, CUFFT_FORWARD);
        // 3. Compute Field Spectra
        addDataToOutput(taperedData[stream_num], signal_field_spectra, stream_num);
        addDataToOutput(taperedNoise[stream_num], noise_field_spectra, stream_num);
        // 4. Compute Power Spectra
        thrust::for_each(thrust::cuda::par_nosync.on(streams[stream_num]),
            thrust::make_zip_iterator(taperedData[stream_num].begin(), taperedNoise[stream_num].begin(), power_spectra.begin()),
            thrust::make_zip_iterator(taperedData[stream_num].end(), taperedNoise[stream_num].end(), power_spectra.end()),
            thrust::make_zip_function(power_functor()));
    }

}

template<typename T>
thrust::host_vector<T> dsp::getCumulativeTrace(const thrust::device_vector<T>* traces)
{
    thrust::device_vector<T> tmp(traces->size(), T(0));
    this->handleError(cudaDeviceSynchronize());
    for (int i = 0; i < num_streams; i++)
        thrust::transform(traces[i].begin(), traces[i].end(), tmp.begin(), tmp.begin(), thrust::plus<T>());
    size_t N = traces->size() / batch_size;
    thrust::host_vector<T> host_trace(N);
    using iter = typename thrust::device_vector<T>::iterator;
    for (size_t j = 0; j < N; ++j) {
        strided_range<iter> tmp_iter(tmp.begin() + j, tmp.end(), N);
        T el = thrust::reduce(tmp_iter.begin(), tmp_iter.end(), T(0), thrust::plus<T>());
        host_trace[j] = el / T(batch_size);
    }
    return host_trace;
}

// Returns the average value
void dsp::getCorrelator(hostvec_c& result)
{
    gpuvec_c c(out[0].size(), tcf(0));
    this->handleError(cudaDeviceSynchronize());
    for (int i = 0; i < num_streams; i++)
        thrust::transform(out[i].begin(), out[i].end(), c.begin(), c.begin(), thrust::plus<tcf>());
    result = c;
}

// Returns the cumulative power
hostvec dsp::getCumulativePower()
{
    return getCumulativeTrace(power);
}

hostvec dsp::getPowerSpectrum()
{
    return getCumulativeTrace(spectrum);
}

hostvec dsp::getPeriodogram()
{
    return getCumulativeTrace(periodogram);
}

hostvec_c dsp::getDataSpectrum()
{
    return getCumulativeTrace(data_fft);
}

hostvec_c dsp::getNoiseSpectrum()
{
    return getCumulativeTrace(noise_fft);
}

// Returns the cumulative field
hostvec_c dsp::getCumulativeField()
{
    return getCumulativeTrace(field);
}

hostvec_c dsp::getCumulativeSubtrData()
{
    return getCumulativeTrace(subtraction_data);
}

hostvec_c dsp::getCumulativeSubtrNoise()
{
    return getCumulativeTrace(subtraction_noise);
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

void dsp::setSubtractionTrace(hostvec_c &trace, hostvec_c& offsets)
{
    subtraction_trace = trace;
    subtraction_offs = offsets;
}

void dsp::getSubtractionTrace(hostvec_c &trace, hostvec_c& offsets)
{
    trace = subtraction_trace;
    offsets = subtraction_offs;
}

void dsp::resetSubtractionTrace()
{
    thrust::fill(subtraction_trace.begin(), subtraction_trace.end(), tcf(0));
    thrust::fill(subtraction_offs.begin(), subtraction_offs.end(), tcf(0));
}

void dsp::setTapers(std::vector<stdvec> h_tapers)
{
    if (h_tapers.size() != K)
        throw std::runtime_error("Tapers number is not equal K");
    if (h_tapers[0].size() != resampled_trace_length)
        throw std::runtime_error("Taper length is not equal resampled_trace_length");
    for (size_t i = 0; i < K; i++)
    {
        hostvec h_taper_batched(resampled_total_length);
        tiled_range<stdvec::iterator> tiled_taper_range(h_tapers[i].begin(), h_tapers[i].end(), batch_size);
        thrust::copy(tiled_taper_range.begin(), tiled_taper_range.end(), h_taper_batched.begin());
        tapers[i] = h_taper_batched;
    }
}

std::vector<hostvec> dsp::getDPSSTapers()
{
    std::vector<hostvec> h_tapers(K);
    for (int i = 0; i < K; i++)
    {
        h_tapers[i] = tapers[i];
    }
    return h_tapers;
}