//
// Created by andrei on 3/27/21.
//

#include "dsp.cuh"
#include "dsp_functors.cuh"
#include <cuda_fp16.h>
#include <cstdio>
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
#include <thrust/async/transform.h>
#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
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
dsp::dsp(size_t len, uint64_t n, double part, int K_, double samplerate) : 
                                       trace_length{static_cast<size_t>(std::round((double)len * part))}, // Length of a signal or noise trace
                                       batch_size{n},                                    // Number of segments in a buffer (same: number of traces in data)
                                       total_length{batch_size * trace_length},
                                       out_size{trace_length * trace_length},
                                       trace1_start{0},       // Start of the signal data
                                       trace2_start{len / 2}, // Start of the noise data
                                       pitch{len}           // Segment length in a buffer
                                       
{
    //firwin.resize(total_length); // GPU memory for the filtering window
    subtraction_trace.resize(total_length);
    subtraction_offs.resize(total_length);
    downconversion_coeffs.resize(total_length);

    // Setup multitaper
    K = K_;

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
        //data_fft_norm[i].resize(total_length);
        subtraction_data[i].resize(total_length);
        noise[i].resize(total_length);
        //noise_fft_norm[i].resize(total_length);
        subtraction_noise[i].resize(total_length);
        power[i].resize(total_length);
        field[i].resize(total_length);
        //out[i].resize(out_size);
        taperedData[i].resize(K * total_length);
        taperedNoise[i].resize(K * total_length);
        data_fft[i].resize(K * total_length);
        noise_fft[i].resize(K * total_length);
        spectrum[i].resize(K * total_length);
        tapers[i].resize(K* total_length);
        //periodogram[i].resize(total_length);

        // Initialize cuFFT plans
        check_cufft_error(cufftPlan1d(&plans[i], trace_length, CUFFT_C2C, batch_size),
                          "Error initializing cuFFT plan\n");
        //check_cufft_error(cufftPlan1d(&multitaper_plans[i], trace_length, CUFFT_C2C, K * batch_size),
        //    "Error initializing cuFFT plan\n");
        cufftCreate(&multitaper_plans[i]);
        size_t workSize;
        long long n[] = { trace_length };
        std::cout << trace_length << std::endl;
        check_cufft_error(cufftXtMakePlanMany(multitaper_plans[i], 1, n,
            NULL, 1, trace_length, CUDA_C_16F,  // Input descriptor
            NULL, 1, trace_length, CUDA_C_16F,  // Output descriptor
            batch_size * K, &workSize, CUDA_C_16F), "Error initializing cuFFT plan for multitaper\n");

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
    hostvec_ch hDownConv(total_length);
    thrust::tabulate(hDownConv.begin(), hDownConv.end(),
        [=] __host__ (int i) -> half2 {
            float t = 0.8 * ovs * static_cast<float>(i % trace_length);
            tcf v = thrust::exp(tcf(0, -2 * pi * frequency * t));
            return make_half2(__float2half(v.real()), __float2half(v.imag()));
        });
    downconversion_coeffs = hDownConv;
}

void dsp::downconvert(gpuvec_ch &data, cudaStream_t& stream)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream), data.begin(), data.end(),
        downconversion_coeffs.begin(), data.begin(), multiply_half2_functor());
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
void dsp::applyDownConversionCalibration(gpuvec_ch& data, cudaStream_t &stream)
{
    auto sync_exec_policy = thrust::cuda::par_nosync.on(stream);
    thrust::transform(sync_exec_policy, data.begin(), data.end(),
        data.begin(), calibration_functor(a_qi, a_qq, c_i, c_q));
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
    applyDownConversionCalibration(noise[stream_num],streams[stream_num]);
    //applyFilter(data[stream_num], firwin, stream_num);
    //applyFilter(noise[stream_num], firwin, stream_num);
    downconvert(data[stream_num], streams[stream_num]);
    downconvert(noise[stream_num], streams[stream_num]);

    subtractDataFromOutput(subtraction_trace, data[stream_num], streams[stream_num]);
    subtractDataFromOutput(subtraction_offs, noise[stream_num], streams[stream_num]);

    addDataToOutput(data[stream_num], subtraction_data[stream_num], streams[stream_num]);
    addDataToOutput(noise[stream_num], subtraction_noise[stream_num], streams[stream_num]);

    calculateField(data[stream_num], noise[stream_num],
        field[stream_num], streams[stream_num]);
    calculatePower(data[stream_num], noise[stream_num],
        power[stream_num], streams[stream_num]);

    calculateMultitaperSpectrum(data[stream_num], noise[stream_num],
        data_fft[stream_num], noise_fft[stream_num], spectrum[stream_num], stream_num);
    //calculatePeriodogram(data_calibrated[stream_num], noise_calibrated[stream_num],
    //    periodogram[stream_num], stream_num);
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
void dsp::convertDataToMillivolts(gpuvec_ch& data, gpubuf& gpu_buf, cudaStream_t &stream)
{
    strided_range<gpubuf::iterator> channelI(gpu_buf.begin(), gpu_buf.end(), 2);
    strided_range<gpubuf::iterator> channelQ(gpu_buf.begin() + 1, gpu_buf.end(), 2);
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
void dsp::addDataToOutput(gpuvec_ch &data, gpuvec_c &output, cudaStream_t& stream)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream), data.begin(), data.end(),
        output.begin(), output.begin(), add_half2_to_tcf_functor());
}

// Subtracts newly processed data from previous data
void dsp::subtractDataFromOutput(gpuvec_ch& data, gpuvec_ch& output, cudaStream_t &stream)
{
    thrust::transform(thrust::cuda::par_nosync.on(stream), output.begin(), output.end(), data.begin(),
        output.begin(), subtract_half2_functor());
}

// Calculates the field from the data in the GPU memory
void dsp::calculateField(gpuvec_ch& data, gpuvec_ch& noise, gpuvec_c& output, cudaStream_t &stream)
{
    thrust::for_each(thrust::cuda::par_nosync.on(stream),
        thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
        thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
        thrust::make_zip_function(field_functor()));
}

// Calculates the power from the data in the GPU memory
void dsp::calculatePower(gpuvec_ch& data, gpuvec_ch& noise, gpuvec& output, cudaStream_t& stream)
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

//void dsp::calculatePeriodogram(gpuvec_c& data, gpuvec_c& noise, gpuvec& output, int stream_num)
//{
//    cufftComplex* cufft_data = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(data.data()));
//    auto cufftstat1 = cufftExecC2C(plans[stream_num], cufft_data, cufft_data, CUFFT_FORWARD);
//    check_cufft_error(cufftstat1, "Error executing cufft");
//
//    cufftComplex* cufft_noise = reinterpret_cast<cufftComplex*>(thrust::raw_pointer_cast(noise.data()));
//    auto cufftstat2 = cufftExecC2C(plans[stream_num], cufft_noise, cufft_noise, CUFFT_FORWARD);
//    check_cufft_error(cufftstat2, "Error executing cufft");
//
//    thrust::for_each(thrust::cuda::par_nosync.on(streams[stream_num]),
//        thrust::make_zip_iterator(data.begin(), noise.begin(), output.begin()),
//        thrust::make_zip_iterator(data.end(), noise.end(), output.end()),
//        thrust::make_zip_function(power_functor()));
//}

void dsp::calculateMultitaperSpectrum(gpuvec_ch& data, gpuvec_ch& noise, gpuvec_c& signal_field_spectra,
    gpuvec_c& noise_field_spectra, gpuvec& power_spectra, int stream_num)
{
    // 1. Windowing the Signal with Tapers
    tiled_range<gpuvec_ch::iterator> tiled_data(data.begin(), data.end(), K);
    tiled_range<gpuvec_ch::iterator> tiled_noise(noise.begin(), noise.end(), K);
    thrust::async::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
        tiled_data.begin(), tiled_data.end(), tapers[stream_num].begin(),
        taperedData[stream_num].begin(), taper_functor());
    thrust::async::transform(thrust::cuda::par_nosync.on(streams[stream_num]),
        tiled_noise.begin(), tiled_noise.end(), tapers[stream_num].begin(),
        taperedNoise[stream_num].begin(), taper_functor());
    // 2. FFT
    half2* cufft_tapered_data = thrust::raw_pointer_cast(taperedData[stream_num].data());
    half2* cufft_tapered_noise = thrust::raw_pointer_cast(taperedNoise[stream_num].data());
    cufftXtExec(multitaper_plans[stream_num], cufft_tapered_data, cufft_tapered_data, CUFFT_FORWARD);
    cufftXtExec(multitaper_plans[stream_num], cufft_tapered_noise, cufft_tapered_noise, CUFFT_FORWARD);
    // 3. Compute Field Spectra
    //addDataToOutput(taperedData[stream_num], signal_field_spectra, streams[stream_num]);
    //addDataToOutput(taperedNoise[stream_num], noise_field_spectra, streams[stream_num]);
    thrust::for_each(thrust::cuda::par_nosync.on(streams[stream_num]),
        thrust::make_zip_iterator(taperedData[stream_num].begin(), taperedNoise[stream_num].begin(), signal_field_spectra.begin(), noise_field_spectra.begin()),
        thrust::make_zip_iterator(taperedData[stream_num].end(), taperedNoise[stream_num].end(), signal_field_spectra.end(), noise_field_spectra.end()),
        thrust::make_zip_function(add_all_functor()));
    // 4. Compute Power Spectra
    thrust::for_each(thrust::cuda::par_nosync.on(streams[stream_num]),
        thrust::make_zip_iterator(taperedData[stream_num].begin(), taperedNoise[stream_num].begin(), power_spectra.begin()),
        thrust::make_zip_iterator(taperedData[stream_num].end(), taperedNoise[stream_num].end(), power_spectra.end()),
        thrust::make_zip_function(power_functor()));

}

// Returns the average value
void dsp::getCorrelator(hostvec_c &result)
{
    gpuvec_c c(out[0].size(), tcf(0));
    this->handleError(cudaDeviceSynchronize());
    for (int i = 0; i < num_streams; i++)
        thrust::transform(out[i].begin(), out[i].end(), c.begin(), c.begin(), thrust::plus<tcf>());
    result = c;
}

template<typename T>
thrust::host_vector<T> dsp::getCumulativeTrace(const thrust::device_vector<T>* traces, size_t M)
{
    thrust::device_vector<T> tmp(traces[0].size(), T(0));
    this->handleError(cudaDeviceSynchronize());
    for (int i = 0; i < num_streams; i++)
        thrust::transform(traces[i].begin(), traces[i].end(), tmp.begin(), tmp.begin(), thrust::plus<T>());
    size_t N = trace_length;
    thrust::host_vector<T> host_trace(N);
    using iter = typename thrust::device_vector<T>::iterator;
    for (size_t j = 0; j < N; ++j) {
        strided_range<iter> tmp_iter(tmp.begin() + j, tmp.end(), N);
        T el = thrust::reduce(tmp_iter.begin(), tmp_iter.end(), T(0), thrust::plus<T>());
        host_trace[j] = el / static_cast<float>(M);
    }
    return host_trace;
}

// Returns the cumulative field
hostvec_c dsp::getCumulativeField()
{
    return getCumulativeTrace(field, batch_size);
}

// Returns the cumulative power
hostvec dsp::getCumulativePower()
{
    return getCumulativeTrace(power, batch_size);
}

hostvec_c dsp::getCumulativeSubtrData()
{
    return getCumulativeTrace(subtraction_data, batch_size);
}

hostvec_c dsp::getCumulativeSubtrNoise()
{
    return getCumulativeTrace(subtraction_noise, batch_size);
}

hostvec dsp::getPowerSpectrum()
{
    return getCumulativeTrace(spectrum, batch_size * K);
}

hostvec dsp::getPeriodogram()
{
    return getCumulativeTrace(periodogram, batch_size);
}

hostvec_c dsp::getDataSpectrum()
{
    return getCumulativeTrace(data_fft, batch_size * K);
}

hostvec_c dsp::getNoiseSpectrum()
{
    return getCumulativeTrace(noise_fft, batch_size * K);
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
    hostvec_ch host_trace(trace.size());
    hostvec_ch host_offsets(trace.size());
    thrust::transform(trace.begin(), trace.end(), host_trace.begin(), tcf_to_half2_functor());
    thrust::transform(offsets.begin(), offsets.end(), host_offsets.begin(), tcf_to_half2_functor());
    subtraction_trace = host_trace;
    subtraction_offs = host_offsets;
}

void dsp::getSubtractionTrace(hostvec_c &trace, hostvec_c& offsets)
{
    gpuvec_c f_trace(subtraction_trace.size());
    gpuvec_c f_offsets(subtraction_offs.size());
    thrust::transform(subtraction_trace.begin(), subtraction_trace.end(), f_trace.begin(), half2_to_tcf_functor());
    thrust::transform(subtraction_offs.begin(), subtraction_offs.end(), f_offsets.begin(), half2_to_tcf_functor());
    trace = f_trace;
    offsets = f_offsets;
}

void dsp::resetSubtractionTrace()
{
    cudaDeviceSynchronize();
    half2 z = make_half2(__float2half(0.f), __float2half(0.f));
    thrust::fill(subtraction_trace.begin(), subtraction_trace.end(), z);
    thrust::fill(subtraction_offs.begin(), subtraction_offs.end(), z);
}

void dsp::setTapers(std::vector<stdvec> h_tapers)
{
    if (h_tapers.size() != K)
        throw std::runtime_error("Tapers number is not equal K");
    if (h_tapers[0].size() != trace_length)
        throw std::runtime_error("Taper length is not equal trace_length");
    hostvec_h taper_fp16(trace_length);
    hostvec_h all_tapers;
    all_tapers.reserve(K * total_length);
    for (auto& h_taper : h_tapers)
    {
        for (size_t j = 0; j < batch_size; j++)
        {
            thrust::transform(h_taper.begin(), h_taper.end(), taper_fp16.begin(), float_to_half_functor());
            all_tapers.insert(all_tapers.end(), taper_fp16.begin(), taper_fp16.end());
        }
    }
    for (int i = 0; i < num_streams; i++)
        tapers[i] = all_tapers;
}

std::vector<hostvec> dsp::getDPSSTapers()
{
    std::vector<hostvec> h_tapers(K);
    gpuvec taper_f(trace_length);
    for (int i = 0; i < K; i++)
    {
        thrust::transform(tapers[0].begin() + i * total_length,
            tapers[0].begin() + (i + 1) * total_length, taper_f.begin(), half_to_float_functor());
        h_tapers[i] = taper_f;
    }
    return h_tapers;
}