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

typedef thrust::complex<float> tcf;
typedef thrust::device_vector<float> gpuvec;
typedef thrust::host_vector<float> hostvec;
typedef thrust::device_vector<tcf> gpuvec_c;
typedef thrust::host_vector<tcf> hostvec_c;
typedef thrust::device_vector<char2> gpubuf;
typedef int8_t *hostbuf;
typedef std::vector<float> stdvec;
typedef std::vector<std::complex<float>> stdvec_c;

class dsp
{
    /* Pointer */
    hostbuf buffer;

    /* Pointers to arrays with data */
    gpubuf gpu_data_buf[num_streams];  // buffers for loading data
    gpubuf gpu_noise_buf[num_streams]; // buffers for loading data

    /* Vectors that hold intermediate data and noise traces */
    gpuvec_c data[num_streams];
    gpuvec_c data_resampled[num_streams];
    gpuvec_c data_fft[num_streams];

    gpuvec_c noise[num_streams];
    gpuvec_c noise_resampled[num_streams];
    gpuvec_c noise_fft[num_streams];

    /* Vectors with results */
    gpuvec power[num_streams];   // arrays for storage of average power
    gpuvec_c field[num_streams]; // arrays for storage of average field
    gpuvec spectrum[num_streams];
    gpuvec periodogram[num_streams];

    /* Filtering windows */
    gpuvec_c firwin;

    /* For accumulation of future subtraction traces */
    gpuvec_c average_data;
    gpuvec_c average_noise;

    /* Subtraction traces */
    gpuvec_c subtraction_data;
    gpuvec_c subtraction_noise;

    /* Downconversion coefficients */
    gpuvec_c downconversion_coeffs;

    /* Useful variables */
    size_t trace_length; // for keeping the length of a trace
    int oversampling;    // determines oversampling after digital filtering
    size_t resampled_trace_length;
    size_t trace1_start, trace2_start, pitch;
    size_t batch_size;   // for keeping the number of segments in data array  // was uint64_t
    size_t total_length; // batch_size * trace_length
    size_t resampled_total_length;
    float scale = 500.f / 128.f; // for conversion into mV

    int semaphore = 0; // for selecting the current stream

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
    cufftHandle welch_plans[num_streams];

    /* cuBLAS required variables */
    cublasHandle_t cublas_handles[num_streams];
    cublasHandle_t cublas_handles2[num_streams];

    /* NVIDIA Performance Primitives required variables */
    NppStreamContext streamContexts[num_streams];

    /* Down-conversion calibration variables */
    float a_qi, a_qq, c_i, c_q;

    /* Welch's method arrays*/
    gpuvec welch_window;
    gpuvec_c replicated_signal[num_streams];
    gpuvec_c replicated_noise[num_streams];
    gpuvec welch_spectrum[num_streams];

    /* Variables for Welch's method applying*/
    int welch_size = 100;
    int welch_overlap = 50;
    int welch_number_of_parts;
    size_t welch_replicated_lenght;

public:
    dsp(size_t len, uint64_t n, double part, int K_, double samplerate, int second_oversampling);

    ~dsp();

    int getTraceLength();

    int getTotalLength();

    void setFirwin(float cutoff_l, float cutoff_r, int oversampling = 1);

    void makeFilterWindow(float cutoff_l, float cutoff_r, gpuvec_c &window, int oversampling = 1);

    void resetOutput();

    void createBuffer(size_t size);

    void deleteBuffer();

    hostbuf getBuffer();

    void compute(const hostbuf buffer_ptr);

    void setWelchWindow();
    void dataReplicationAndWindowing(gpuvec_c &data, gpuvec_c &replication, int stream_num);
    void Welch(gpuvec_c &signal, gpuvec_c &noise, gpuvec &output, int stream_num);
    hostvec getWelchSpectrum();

    hostvec getCumulativePower();

    hostvec_c getCumulativeField();

    std::pair<hostvec_c, hostvec_c> getCumulativeSubtrData();

    void setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q);

    void setSubtractionTraces(hostvec_c &trace, hostvec_c &offsets);

    std::pair<hostvec_c, hostvec_c> getSubtractionTraces();

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
    void handleError(cudaError_t error);

    void switchStream() { semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0; };

    void loadDataToGPUwithPitchAndOffset(const hostbuf buffer_ptr,
                                         gpubuf &gpu_buf, size_t pitch, size_t offset, int stream_num);

    void convertDataToMillivolts(gpuvec_c &data, const gpubuf &gpu_buf, const cudaStream_t &stream);

    void downconvert(gpuvec_c &data, int stream_num);

    void preprocess();

    void applyDownConversionCalibration(gpuvec_c &data, cudaStream_t &stream);

    void addDataToOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void subtractDataFromOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void applyFilter(gpuvec_c &data, const gpuvec_c &window, int stream_num);

    void calculateField(const gpuvec_c &data, const gpuvec_c &noise, gpuvec_c &output, const cudaStream_t &stream);

    void resample(const gpuvec_c &traces, gpuvec_c &resampled_traces, const cudaStream_t &stream);

    void resampleFFT(gpuvec_c &traces, gpuvec_c &resampled_traces, const int &stream_num);

    void normalize(gpuvec_c &data, float coeff, int stream_num);

    void calculatePower(const gpuvec_c &data, const gpuvec_c &noise, gpuvec &output, const cudaStream_t &stream);

    void calculatePeriodogram(gpuvec_c &data, gpuvec_c &noise, gpuvec &output, int stream_num);

    void calculateMultitaperSpectrum(const gpuvec_c &data, const gpuvec_c &noise, gpuvec_c &signal_field_spectra,
                                     gpuvec_c &noise_field_spectra, gpuvec &power_spectra, int stream_num);

    template <typename T>
    thrust::host_vector<T> getCumulativeTrace(const thrust::device_vector<T> *traces);

    template <typename T>
    thrust::device_vector<T> sumOverStreams(const thrust::device_vector<T> *traces);

    template <typename T>
    thrust::host_vector<T> sumOverBatch(const thrust::device_vector<T> &traces);
};

#endif // CPPMEASUREMENT_DSP_CUH
