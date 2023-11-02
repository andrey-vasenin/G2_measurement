//
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
// typedef thrust::host_vector<int8_t, thrust::mr::stateless_resource_allocator<int8_t,
//     thrust::system::cuda::universal_host_pinned_memory_resource> > hostbuf;
// typedef hostbuf::iterator hostbuf_iter_t;
typedef std::vector<float> stdvec;
typedef std::vector<std::complex<float>> stdvec_c;

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

class dsp
{
    /* Pointers to arrays with data */
    gpubuf gpu_data_buf[num_streams];  // buffers for loading data
    gpubuf gpu_noise_buf[num_streams]; // buffers for loading data
    gpuvec_c data[num_streams];
    gpuvec_c data_resampled[num_streams];
    gpuvec_c data_fft[num_streams];
    gpuvec_c subtraction_data[num_streams];

    gpuvec_c noise[num_streams];
    gpuvec_c noise_resampled[num_streams];
    gpuvec_c noise_fft[num_streams];
    gpuvec_c subtraction_noise[num_streams];

    gpuvec power[num_streams];   // arrays for storage of average power
    gpuvec_c field[num_streams]; // arrays for storage of average field
    gpuvec_c out[num_streams];
    gpuvec spectrum[num_streams];
    gpuvec periodogram[num_streams];

    /* Filtering window */
    gpuvec_c firwin;

    /* Subtraction trace */
    gpuvec_c subtraction_trace;
    gpuvec_c subtraction_offs;

    /* Downconversion coefficients */
    gpuvec_c downconversion_coeffs;

private:
    /* Useful variables */
    size_t trace_length; // for keeping the length of a trace
    int oversampling;    // determines oversampling after digital filtering
    size_t resampled_trace_length;
    size_t trace1_start, trace2_start, pitch;
    size_t batch_size;   // for keeping the number of segments in data array  // was uint64_t
    size_t total_length; // batch_size * trace_length
    size_t resampled_total_length;
    size_t out_size;
    int semaphore = 0;           // for selecting the current stream
    float scale = 500.f / 128.f; // for conversion into mV

    /* Multitaper method properties */
    int K; // number of tapers
    std::vector<gpuvec> tapers;
    gpuvec_c taperedData[num_streams];
    gpuvec_c taperedNoise[num_streams];
    cufftHandle resampled_plans[num_streams];

    /* Streams' arrays */
    cudaStream_t streams[num_streams];

    /* cuFFT required variables */
    cufftHandle plans[num_streams];

    /* cuBLAS required variables */
    cublasHandle_t cublas_handles[num_streams];
    cublasHandle_t cublas_handles2[num_streams];

    /* NVIDIA Performance Primitives required variables */
    NppStreamContext streamContexts[num_streams];

    /* Down-conversion calibration variables */
    float a_qi, a_qq, c_i, c_q;

public:
    dsp(size_t len, uint64_t n, double part, int K_, double samplerate, int second_oversampling);

    ~dsp();

    int getTraceLength();

    int getTotalLength();

    int getOutSize();

    void setFirwin(float cutoff_l, float cutoff_r, int oversampling = 1);

    void resetOutput();

    void compute(const hostbuf buffer_ptr);

    hostvec getCumulativePower();

    hostvec_c getCumulativeField();

    hostvec_c getCumulativeSubtrData();

    hostvec_c getCumulativeSubtrNoise();

    hostvec_c getCorrelator();

    void setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q);

    void setSubtractionTrace(hostvec_c &trace, hostvec_c &offsets);

    void getSubtractionTrace(hostvec_c &trace, hostvec_c &offsets);

    void resetSubtractionTrace();

    void setIntermediateFrequency(float frequency, int oversampling);

    void setAmplitude(int ampl);

    void setTapers(std::vector<stdvec> h_tapers);

    std::vector<hostvec> getDPSSTapers();

    hostvec getPowerSpectrum();
    hostvec_c getDataSpectrum();
    hostvec_c getNoiseSpectrum();
    hostvec getPeriodogram();

protected:
    template <typename T>
    thrust::host_vector<T> getCumulativeTrace(const thrust::device_vector<T> *traces);

    void handleError(cudaError_t error);

    void switchStream() { semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0; };

    void loadDataToGPUwithPitchAndOffset(const hostbuf buffer_ptr,
                                         gpubuf &gpu_buf, size_t pitch, size_t offset, int stream_num);

    void convertDataToMillivolts(gpuvec_c &data, const gpubuf &gpu_buf, const cudaStream_t &stream);

    void downconvert(gpuvec_c &data, int stream_num);

    void applyDownConversionCalibration(gpuvec_c &data, cudaStream_t &stream);

    void addDataToOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void subtractDataFromOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void applyFilter(gpuvec_c &data, const gpuvec_c &window, int stream_num);

    void calculateField(const gpuvec_c &data, const gpuvec_c &noise, gpuvec_c &output, const cudaStream_t &stream);

    void resample(const gpuvec_c &traces, gpuvec_c &resampled_traces, const cudaStream_t &stream);

    void resampleFFT(gpuvec_c &traces, gpuvec_c &resampled_traces, const int &stream_num);

    void normalize(gpuvec_c &data, float coeff, int stream_num);

    void calculatePower(const gpuvec_c &data, const gpuvec_c &noise, gpuvec &output, const cudaStream_t &stream);

    void calculateG1(gpuvec_c &data, gpuvec_c &noise, gpuvec_c &output, cublasHandle_t &handle);

    void calculatePeriodogram(gpuvec_c &data, gpuvec_c &noise, gpuvec &output, int stream_num);

    void calculateMultitaperSpectrum(const gpuvec_c &data, const gpuvec_c &noise, gpuvec_c &signal_field_spectra,
                                     gpuvec_c &noise_field_spectra, gpuvec &power_spectra, int stream_num);
};

#endif // CPPMEASUREMENT_DSP_CUH
