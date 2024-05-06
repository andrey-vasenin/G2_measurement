//
// Created by andrei on 3/27/21.
//

#pragma once
#ifndef DSP_H
#define DSP_H

#include <nppdefs.h>
#include <array>
#include <vector>
#include <complex>
#include <cufft.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>
#include "dsp_types.cuh"
#include "dsp_helpers.cuh"
#include "dsp_micromodules.cuh"

const int num_streams = 2;
const int cal_mat_size = 16;
const int cal_mat_side = 4;
const int num_channels = 1; // number of used digitizer channels

class dsp
{
    /* Pointer */
    hostbuf buffer;

    /* Pointers to arrays with data */
    gpubuf gpu_data_buf[num_streams];  // buffers for loading data
    gpubuf gpu_noise_buf[num_streams]; // buffers for loading data
    gpuvec_c data[num_streams];
    gpuvec_c noise[num_streams];
    gpuvec_c data_resampled[num_streams];
    gpuvec_c noise_resampled[num_streams];
    gpuvec_c subtraction_data;
    gpuvec_c subtraction_noise;
    gpuvec_c data_sideband1[num_streams];
    gpuvec_c data_sideband2[num_streams];
    gpuvec_c data_tmp[num_streams];
    gpuvec_c noise_sideband1[num_streams];
    gpuvec_c noise_sideband2[num_streams];
    gpuvec_c noise_tmp[num_streams];
    gpuvec_c data_central_peak[num_streams];
    gpuvec_c noise_central_peak[num_streams];

    gpuvec_c interference_out[num_streams];
    gpuvec_c g1_filt[num_streams];
    gpuvec_c g1_filt_cross_segment[num_streams];
    gpuvec_c g1_filt_conj[num_streams];
    gpuvec_c g1_filt_conj_cross_segment[num_streams];
    gpuvec_c g1_without_cp[num_streams];
    gpuvec_c g1_without_cp_cross_segment[num_streams];
    gpuvec_c g2_filt[num_streams];
    gpuvec g2_filt_cross_segment[num_streams];

    gpuvec PSD_sideband1[num_streams];
    gpuvec PSD_sideband2[num_streams];
    gpuvec PSD_rayleigh[num_streams];
    gpuvec PSD_total[num_streams];
    gpuvec PSD_sidebands_product[num_streams];

    gpuvec power_short[num_streams];
    gpuvec_c data_product[num_streams];
    gpuvec_c noise_product[num_streams];

    /* Filtering windows */
    gpuvec_c firwin;
    gpuvec_c center_peak_win;
    gpuvec_c corr_firwin1;
    gpuvec_c corr_firwin2;

    /* Average traces */
    gpuvec_c average_data;
    gpuvec_c average_noise;

    /* Downconversion coefficients */
    gpuvec_c downconversion_coeffs;
    gpuvec_c corr_downconversion_coeffs1;
    gpuvec_c corr_downconversion_coeffs2;

    std::array<std::unique_ptr<micromodule>, num_streams> modules;

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

    const cuComplex alpha_c = make_cuComplex(1, 0);
    const cuComplex beta_data_c = make_cuComplex(1, 0);
    const cuComplex beta_noise_c = make_cuComplex(-1, 0);
    const float alpha_f = 1.0;
    const float beta_f = 1.0;
    cublasOperation_t op_t = CUBLAS_OP_T;
    cublasOperation_t op_c = CUBLAS_OP_C;

    /* Streams' arrays */
    // std::array<std::shared_ptr<cudaStream_t>, num_streams> streams;
    cudaStream_t streams[num_streams];

    /* cuFFT required variables */
    // std::array<std::shared_ptr<cufftHandle>, num_streams> plans;
    // std::array<std::shared_ptr<cufftHandle>, num_streams> plans_resampled;
    cufftHandle plans[num_streams];
    cufftHandle plans_resampled[num_streams];

    /* cuBLAS required variables */
    std::array<std::shared_ptr<cublasHandle_t>, num_streams> cublas_handles;
    // cublasHandle_t cublas_handles[num_streams];

    /* NVIDIA Performance Primitives required variables */
    NppStreamContext streamContexts[num_streams];

    /* Down-conversion calibration variables */
    float a_qi, a_qq, c_i, c_q;

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

    std::pair<hostvec_c, hostvec_c> getAverageData();

    std::pair<hostvec_c, hostvec_c> getG1FiltResult();

    std::pair<hostvec_c, hostvec_c> getG1FiltConjResult();

    std::pair<hostvec_c, hostvec_c> getG1WithoutCPResult();

    std::pair<hostvec_c, hostvec> getG2FilteredResult();

    std::tuple<hostvec, hostvec, hostvec, hostvec, hostvec> getPSDResults();

    hostvec_c getInterferenceResult();

    void setDownConversionCalibrationParameters(float r, float phi, float offset_i, float offset_q);

    void setSubtractionTrace(hostvec_c trace[num_channels]);

    void getSubtractionTrace(std::vector<stdvec_c> &trace);

    void resetSubtractionTrace();

    void createBuffer(size_t size);

    void deleteBuffer();

    hostbuf getBuffer();

    void setIntermediateFrequency(float frequency, int oversampling);

    void setCorrDowncovertCoeffs(float freq1, float freq2, int oversampling);

    void setAmplitude(int ampl);

    std::vector<hostvec_c> getAllFirwins();

    template <typename T>
    static thrust::host_vector<T> getCumulativeTrace(const thrust::device_vector<T> *traces, size_t batch_size);

    template <typename T>
    static thrust::device_vector<T> sumOverStreams(const thrust::device_vector<T> *traces);

    template <typename T>
    static thrust::host_vector<T> sumOverBatch(const thrust::device_vector<T> &trace, size_t batch_size);

    template <typename VectorT, typename T>
    static void divideBy(VectorT &trace, T div);

    static void applyFilter(gpuvec_c &data, gpuvec_c &window, size_t size, cudaStream_t &stream, cufftHandle &plan);

    static void handleError(cudaError_t error);

protected:
    void switchStream()
    {
        semaphore = (semaphore < (num_streams - 1)) ? semaphore + 1 : 0;
    };

    void loadDataToGPUwithPitchAndOffset(const hostbuf buffer_ptr,
                                         gpubuf &gpu_buf, size_t pitch, size_t offset, int stream_num);

    void convertDataToMillivolts(gpuvec_c &data, const gpubuf &gpu_buf, const cudaStream_t &stream);

    void downconvert(gpuvec_c &data, int stream_num);

    void applyDownConversionCalibration(gpuvec_c &data, cudaStream_t &stream);

    void addDataToOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void subtractDataFromOutput(const gpuvec_c &data, gpuvec_c &output, int stream_num);

    void extractSideband(gpuvec_c &src, gpuvec_c &dst, gpuvec_c &filterwin, int stream_num);

    void copyData(gpuvec_c &source, gpuvec_c &dist, cudaStream_t &stream);

    void resample(const gpuvec_c &traces, gpuvec_c &resampled_traces, const cudaStream_t &stream);

    void normalize(gpuvec_c &data, float coeff, int stream_num);

    void calculatePower(gpuvec_c &data, gpuvec_c &noise, gpuvec &output, cudaStream_t &stream);

    void calculatePSD(gpuvec_c &data, gpuvec_c &noise, gpuvec &output, int stream_num);

    void calculateSignalsProduct(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &output, cudaStream_t &stream);

    void calculateSidebandsProductPSD(gpuvec_c &sideband1, gpuvec_c &sideband2,
                                      gpuvec_c &noise1, gpuvec_c &noise2, gpuvec &psd,
                                      int stream_num);

    void calculateG1(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &noise1, gpuvec_c &noise2, gpuvec_c &output, cublasHandle_t &handle, cublasOperation_t &op);

    void calculateG1cs(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &noise1, gpuvec_c &noise2,
                       gpuvec_c &data1_short, gpuvec_c &noise_short, gpuvec_c &output, gpuvec_c &output_cs, cublasHandle_t &handle, cublasOperation_t &op, const cudaStream_t &stream);

    void calculateG2(gpuvec &power1, gpuvec &power2, gpuvec &output, cublasHandle_t &handle);

    void calculateG2cs(gpuvec_c &data_1, gpuvec_c &data_2, gpuvec_c &cross_power, gpuvec_c &cross_power_short, gpuvec_c &output_one_segment,
                       gpuvec_c &output_cross_segment, const cudaStream_t &stream, cublasHandle_t &handle);

    void calculateG2csAlt(gpuvec &power1, gpuvec &power2, gpuvec &power_short,
                          gpuvec &output_one_segment, gpuvec &output_cross_segment, const cudaStream_t &stream, cublasHandle_t &handle);

    void calculateInterference(gpuvec_c &data1, gpuvec_c &data2, gpuvec_c &noise1, gpuvec_c &noise2, gpuvec_c &output, int stream_num);

    static void cufftPlanDeleter(cufftHandle *ptr);
    static void cublasDeleter(cublasHandle_t *ptr);
    static void streamDeleter(cudaStream_t *ptr);
};

#endif // DSP_H