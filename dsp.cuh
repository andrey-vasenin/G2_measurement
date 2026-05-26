//
// Created by andrei on 3/27/21.
//

#ifndef CPPMEASUREMENT_DSP_CUH
#define CPPMEASUREMENT_DSP_CUH

#include <nppdefs.h>
#include <cuComplex.h>
#include <vector>
#include <complex>
#include <tuple>
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
const int num_channels = 2; // maximum number of complex fields

typedef thrust::complex<float> tcf;
typedef thrust::device_vector<float> gpuvec;
typedef thrust::host_vector<float> hostvec;
typedef thrust::device_vector<tcf> gpuvec_c;
typedef thrust::host_vector<tcf> hostvec_c;
typedef thrust::device_vector<char4> gpubuf;
typedef thrust::device_vector<char2> gpubuf_one;
typedef int8_t *hostbuf;
typedef std::vector<float> stdvec;
typedef std::vector<std::complex<float>> stdvec_c;

enum class ResultMode
{
    AverageOnly,
    AverageG1,
    AllCorrelators
};

inline const char *resultModeName(ResultMode mode)
{
    switch (mode)
    {
    case ResultMode::AverageOnly:
        return "average";
    case ResultMode::AverageG1:
        return "average_g1";
    case ResultMode::AllCorrelators:
        return "all_correlators";
    default:
        return "unknown";
    }
}

enum class ChannelLayout
{
    OneComplexField,
    TwoComplexFields
};

inline const char *channelLayoutName(ChannelLayout layout)
{
    switch (layout)
    {
    case ChannelLayout::OneComplexField:
        return "one_complex";
    case ChannelLayout::TwoComplexFields:
        return "two_complex";
    default:
        return "unknown";
    }
}

inline int complexFieldCount(ChannelLayout layout)
{
    return (layout == ChannelLayout::OneComplexField) ? 1 : 2;
}

inline int physicalChannelCount(ChannelLayout layout)
{
    return (layout == ChannelLayout::OneComplexField) ? 2 : 4;
}

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

struct AverageState
{
    gpubuf gpu_data_buf[num_streams];
    gpubuf_one gpu_data_buf_one[num_streams];
    gpuvec_c data1[num_streams];
    gpuvec_c data2[num_streams];
    gpuvec_c data1_resampled[num_streams];
    gpuvec_c data2_resampled[num_streams];
    gpuvec_c subtraction_data1[num_streams];
    gpuvec_c subtraction_data2[num_streams];
    gpuvec_c subtraction_trace1;
    gpuvec_c subtraction_trace2;
    gpuvec_c tmp1;
    gpuvec_c tmp2;
    float2 *s21_sum1 = nullptr;
    float2 *s21_sum2 = nullptr;
};

struct G1State
{
    gpuvec_c data2_resampled_conj[num_streams];
    gpuvec_c g1[num_streams];
    cublasHandle_t cublas_handles[num_streams];
};

struct AllCorrelatorState
{
    gpuvec_c data1_resampled_conj[num_streams];
    gpuvec_c g1_annihilation[num_streams];
    gpuvec_c g1_creation[num_streams];
    gpuvec_c g1_reordered[num_streams];
    gpuvec_c cross_power[num_streams];
    gpuvec_c cross_spectrum[num_streams];
    gpuvec_c tmp_cross;
    cufftHandle corr_plans[num_streams];
};

class dsp
{
    /* Pointer */
    hostbuf buffer;

    AverageState average_state;
    G1State g1_state;
    AllCorrelatorState all_state;

    /* Filtering windows */
    gpuvec_c firwin;

    /* Downconversion coefficients */
    gpuvec_c downconversion_coeffs;
    gpuvec_c corr_downconversion_coeffs1;
    gpuvec_c corr_downconversion_coeffs2;

private:
    /* Useful variables */
    size_t trace_length; // for keeping the length of a trace
    int oversampling;    // determines oversampling after digital filtering
    size_t resampled_trace_length;
    size_t pitch;
    size_t batch_size;   // for keeping the number of segments in data array  // was uint64_t
    size_t total_length; // batch_size * trace_length
    size_t resampled_total_length;
    size_t out_size;
    ResultMode result_mode;
    ChannelLayout channel_layout;
    int complex_fields;
    int physical_channels;
    int semaphore = 0;           // for selecting the current stream
    float scale = 500.f / 128.f; // for conversion into mV // max int8 is 127

    const cuComplex alpha = make_cuComplex(1, 0);
    const cuComplex beta = make_cuComplex(1, 0);
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasOperation_t op_t = CUBLAS_OP_T;

    /* Streams' arrays */
    cudaStream_t streams[num_streams];
    cudaEvent_t input_copy_done[num_streams];

    /* cuFFT required variables */
    cufftHandle plans[num_streams];

    /* NVIDIA Performance Primitives required variables */
    NppStreamContext streamContexts[num_streams];

    /* Down-conversion calibration variables */
    float a_qi[num_channels], a_qq[num_channels], c_i[num_channels], c_q[num_channels];

public:
    dsp(size_t len, uint64_t n, double samplerate, int second_oversampling, ResultMode mode,
        ChannelLayout layout);

    ~dsp();

    const char *getResultModeName() const { return resultModeName(result_mode); }

    const char *getChannelLayoutName() const { return channelLayoutName(channel_layout); }

    int getComplexFieldCount() const { return complex_fields; }

    int getPhysicalChannelCount() const { return physical_channels; }

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

    void makeFilterWindow(float cutoff_l, float cutoff_r, gpuvec_c &window, size_t trace_len, size_t total_len, int oversampling = 1);

    void resetOutput();

    int compute(const hostbuf buffer_ptr);

    void waitInputCopy(int stream_num);

    void synchronize();

    std::vector<hostvec_c> getCumulativeSubtrData();
  
    hostvec_c getG1Result();

    std::tuple<hostvec_c, hostvec_c, hostvec_c> getG1OtherResults();

    std::pair<stdvec_c, stdvec_c> getAverageField();

    std::pair<std::complex<float>, std::complex<float>> getS21();

    hostvec_c getCrossPower();

    hostvec_c getCrossSpectrum();
    
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
    bool hasSecondField() const;

    bool hasG1() const;

    bool hasAllCorrelators() const;

    void requireG1(const char *getter_name) const;

    void requireAllCorrelators(const char *getter_name) const;

    template <typename T>
    thrust::host_vector<T> getCumulativeTrace(const thrust::device_vector<T> *traces, const T divisor);

    void handleError(cudaError_t error);

    void switchStream() { semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0; };

    void copyDataFromBuffer(const hostbuf buffer_ptr, 
                                        gpubuf &dst, int stream_num);

    void copyDataFromBuffer(const hostbuf buffer_ptr,
                                        gpubuf_one &dst, int stream_num);

    void splitAndConvertDataToMillivolts(gpuvec_c &data_left, gpuvec_c &data_right, const gpubuf &gpu_buf, const cudaStream_t &stream);

    void splitAndConvertDataToMillivolts(gpuvec_c &data, const gpubuf_one &gpu_buf, const cudaStream_t &stream);

    void downconvert(gpuvec_c &data, int stream_num);

    void applyDownConversionCalibration(gpuvec_c &data, cudaStream_t &stream, int channel_num);

    void addDataToOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void subtractDataFromOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void applyFilter(gpuvec_c &data, const gpuvec_c &window, int stream_num, size_t length, cufftHandle &plan);

    void calculateFFT(gpuvec_c &data, int stream_num, int direction, cufftHandle &plan);

    void resample(const gpuvec_c &traces, gpuvec_c &resampled_traces, const cudaStream_t &stream);

    void calculateG1gemm(gpuvec_c& data1, gpuvec_c& data2, gpuvec_c& output, cublasHandle_t &handle, cublasOperation_t &op_1, cublasOperation_t &op_2);

    void calculateG2gemm(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &cross_power, gpuvec_c &output, const cudaStream_t &stream, cublasHandle_t &handle);
};

#endif // CPPMEASUREMENT_DSP_CUH
