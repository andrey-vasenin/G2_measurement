//
// Created by andrei on 3/27/21.
//

#ifndef CPPMEASUREMENT_DSP_CUH
#define CPPMEASUREMENT_DSP_CUH

#include <nppdefs.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <cufft.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>

const int num_streams = 2;
const int cal_mat_size = 16;
const int cal_mat_side = 4;
const int num_channels = 2; // number of used digitizer channels

typedef thrust::complex<float> tcf;
typedef thrust::device_vector<float> gpuvec;
typedef thrust::host_vector<float> hostvec;
typedef thrust::device_vector<tcf> gpuvec_c;
typedef thrust::host_vector<tcf> hostvec_c;
typedef thrust::device_vector<char4> gpubuf;
typedef int8_t *hostbuf;
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
    /* Pointer */
    hostbuf buffer;

    /* Pointers to arrays with data */
    gpubuf gpu_data_buf[num_streams];  // buffers for loading data
    gpuvec_c data1[num_streams];
    gpuvec_c data2[num_streams];
    gpuvec_c data1_resampled[num_streams];
    gpuvec_c data2_resampled[num_streams];
    gpuvec_c subtraction_data1[num_streams];
    gpuvec_c subtraction_data2[num_streams];
    gpuvec_c data_for_correlation1[num_streams];
    gpuvec_c data_for_correlation2[num_streams];
    gpuvec_c data_without_central_peak1[num_streams];
    gpuvec_c data_without_central_peak2[num_streams];

    gpuvec_c interference_out[num_streams];
    gpuvec_c g1_cross_out[num_streams];
    gpuvec_c g1_filt_conj[num_streams];
    gpuvec_c g1_filt[num_streams];
    gpuvec_c g2_out[num_streams];
    gpuvec_c g2_out_cross_segment[num_streams];
    gpuvec_c g2_out_filtered[num_streams];
    gpuvec_c g2_out_filtered_cross_segment[num_streams];
    gpuvec_c cross_power[num_streams];
    gpuvec_c cross_power_short[num_streams];
    gpuvec_c power1[num_streams];
    gpuvec_c power2[num_streams];
    gpuvec_c power_short[num_streams];

    /* Filtering windows */
    gpuvec_c firwin;
    gpuvec_c center_peak_win;
    gpuvec_c corr_firwin1;
    gpuvec_c corr_firwin2;

    /* Subtraction traces */
    gpuvec_c subtraction_trace1;
    gpuvec_c subtraction_trace2;

    /* Downconversion coefficients */
    gpuvec_c downconversion_coeffs;
    gpuvec_c corr_downconversion_coeffs1;
    gpuvec_c corr_downconversion_coeffs2;

private:
    /* Useful variables */
    size_t trace_length; // for keeping the length of a trace
    int oversampling;    // determines oversampling after digital filtering
    size_t resampled_trace_length;
    size_t trace1_start, trace2_start, pitch;
    size_t inter_buffer_size;
    size_t batch_size;   // for keeping the number of segments in data array  // was uint64_t
    size_t total_length; // batch_size * trace_length
    size_t resampled_total_length;
    size_t out_size;
    int semaphore = 0;           // for selecting the current stream
    float scale = 500.f / 128.f; // for conversion into mV // max int8 is 127

    const cuComplex alpha = make_cuComplex(1, 0);
    const cuComplex beta = make_cuComplex(1, 0);
    const float beta_float = 1.0;
    cublasOperation_t op_t = CUBLAS_OP_T;
    cublasOperation_t op_c = CUBLAS_OP_C;

    /* Streams' arrays */
    cudaStream_t streams[num_streams];

    /* cuFFT required variables */
    cufftHandle plans[num_streams];
    cufftHandle corr_plans[num_streams];

    /* cuBLAS required variables */
    cublasHandle_t cublas_handles[num_streams];

    /* NVIDIA Performance Primitives required variables */
    NppStreamContext streamContexts[num_streams];

    /* Down-conversion calibration variables */
    float a_qi[num_channels], a_qq[num_channels], c_i[num_channels], c_q[num_channels];

public:
    dsp(size_t len, uint64_t n, double part, double samplerate, int second_oversampling);

    ~dsp();

    int getTraceLength();

    int getTotalLength();

    int getOutSize();

    int getResampledTraceLength()
    {
        return resampled_trace_length;
    };

    int getResampledTotalLength()
    {
        return resampled_total_length;
    }

    void setFirwin(float cutoff_l, float cutoff_r, int dig_oversampling = 1);
    void setFirwin(hostvec_c window);

    void setCentralPeakWin(float cutoff_l, float cutoff_r, int dig_oversampling = 1);
    void setCentralPeakWin(hostvec_c window);

    void setCorrelationFirwin(std::pair<float, float> cutoff_1, std::pair<float, float> cutoff_2, int dig_oversampling = 1);
    void setCorrelationFirwin(hostvec_c window1, hostvec_c window2);

    void makeFilterWindow(float cutoff_l, float cutoff_r, gpuvec_c &window, size_t trace_len, size_t total_len, int oversampling = 1);

    void resetOutput();

    void compute(const hostbuf buffer_ptr);

    std::vector<hostvec_c> getCumulativeSubtrData();
  
    hostvec_c getCumulativeCorrelator(gpuvec_c g_out[4]);

    hostvec_c getG1CrossResult();

    hostvec_c getG1FiltResult();

    hostvec_c getG1FiltConjResult();

    hostvec_c getG2FullResult();

    hostvec_c getG2CrossSegmentResult();

    hostvec_c getG2FilteredResult();

    hostvec_c getG2FilteredCrossSegmentResult();

    hostvec_c getInterferenceRsult();

    void setDownConversionCalibrationParameters(int channel_num, float r, float phi, float offset_i, float offset_q);

    void setSubtractionTrace(hostvec_c trace[num_channels]);

    void getSubtractionTrace(std::vector<stdvec_c> &trace);

    void resetSubtractionTrace();

    void createBuffer(size_t size);

    void deleteBuffer();

    hostbuf getBuffer();

    void setIntermediateFrequency(float frequency, int oversampling);

    void setCorrDowncovertCoeffs(float freq1, float freq2, int oversampling);

    void setAmplitude(int ampl);

protected:
    template <typename T>
    thrust::host_vector<T> getCumulativeTrace(const thrust::device_vector<T> *traces, const T divisor);

    void handleError(cudaError_t error);

    void switchStream() { semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0; };

    void copyDataFromBuffer(const hostbuf buffer_ptr, 
                                        gpubuf &dst, int stream_num);

    void splitAndConvertDataToMillivolts(gpuvec_c &data_left, gpuvec_c &data_right, const gpubuf &gpu_buf, const cudaStream_t &stream);

    void downconvert(gpuvec_c &data, int stream_num);

    void calculateInterference(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &output, int stream_num);

    void applyDownConversionCalibration(gpuvec_c &data, cudaStream_t &stream, int channel_num);

    void addDataToOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void subtractDataFromOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void applyFilter(gpuvec_c &data, const gpuvec_c &window, int stream_num, size_t length, cufftHandle &plan);

    void calculateFFT(gpuvec_c &data, int stream_num, int direction, cufftHandle &plan);

    void applyFilterAlt(gpuvec_c &fftdata, const gpuvec_c &window, int stream_num, size_t length, cufftHandle &plan);

    void copyData(gpuvec_c &source, gpuvec_c &dist, cudaStream_t &stream);

    void resample(const gpuvec_c &traces, gpuvec_c &resampled_traces, const cudaStream_t &stream);

    void normalize(gpuvec_c &data, float coeff, int stream_num);

    void calculateG1(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &output, cublasHandle_t &handle);

    void calculateG1gemm(gpuvec_c& data1, gpuvec_c& data2, gpuvec_c& output, cublasHandle_t &handle, cublasOperation_t &op);

    void calculateG2(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &cross_power, gpuvec_c &output, const cudaStream_t &stream, cublasHandle_t &handle);

    void calculateG2gemm(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &cross_power, gpuvec_c &output, const cudaStream_t &stream, cublasHandle_t &handle);

    void calculateG2New(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &cross_power, gpuvec_c &cross_power_short, gpuvec_c &output_one_segment, 
                        gpuvec_c &output_cross_segment,const cudaStream_t &stream, cublasHandle_t &handle);

    void calculateG2Alt(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &power1, gpuvec_c &power2, gpuvec_c &power_short, 
                        gpuvec_c &output_one_segment, gpuvec_c &output_cross_segment, const cudaStream_t &stream, cublasHandle_t &handle);
};

#endif // CPPMEASUREMENT_DSP_CUH
