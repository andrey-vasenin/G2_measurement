//
// Created by andrei on 4/13/21.
//
#pragma once
#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include "digitizer.h"
#include "dsp.cuh"
#include "pinned_allocator.cuh"
#include "yokogawa_gs210.h"

// namespace py = pybind11;

using corr_c = std::vector<std::vector<std::complex<double>>>;
using trace_c = std::vector<std::complex<double>>;
using corr_r = std::vector<std::vector<double>>;
using trace_r = std::vector<double>;

class Measurement
{
private:
    std::unique_ptr<yokogawa_gs210> coil;
    std::unique_ptr<Digitizer> dig;
    std::unique_ptr<dsp> processor;
    size_t segment_size;
    uint64_t segments_count;
    uint64_t batch_size;
    size_t notify_size;
    uint64_t iters_num;
    uint64_t iters_done;
    double sampling_rate;
    hostbuf buffer;

    int second_ovs;

    float offset_current = 0.f;
    float working_current = 0.f;

    float max = 0.f;

    thrust::host_vector<int8_t> test_input;

    proc_t func;
    proc_t func_ult_calib;

public:
    Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch, double part,
                int second_oversampling, const char *coil_address);

    Measurement(std::unique_ptr<Digitizer> dig_, uint64_t averages, uint64_t batch, double part,
                int second_oversampling, const char *coil_address);

    Measurement(uint64_t averages, uint64_t batch, long segment, double part, int dig_oversampling,
                int second_oversampling);

    void setDigParameters();

    void setAmplitude(int ampl);

    void setCurrents(float wc, float oc);

    void setAveragesNumber(uint64_t averages);

    ~Measurement();

    void reset();

    void resetOutput();

    void free();

    void setCalibration(float r, float phi, float offset_i, float offset_q);

    void setFirwin(float left_cutoff, float right_cutoff);
    void setFirwin(const stdvec_c window);

    void setAdditionalFirwins(std::vector<std::pair<float, float>> cutoffs);

    // void setCentralPeakWin(float left_cutoff, float right_cutoff);
    // void setCentralPeakWin(const stdvec_c window);

    // void setCorrelationFirwin(std::pair<float, float> cutoff_1, std::pair<float, float> cutoff_2);
    // void setCorrelationFirwin(const stdvec_c window1, const stdvec_c window2);

    void setIntermediateFrequency(float frequency);

    void setCorrDowncovertCoeffs(float freq1, float freq2);

    void measure();

    void asyncCurrentSwitch();

    void measureWithCoil();

    void measureTest();

    void setTestInput(const std::vector<int8_t> &input);

    std::pair<corr_c, corr_c> getG1Filt();

    std::pair<corr_c, corr_c> getG1FiltConj();

    std::pair<corr_c, corr_c> getG1WithoutCP();

    std::pair<corr_c, corr_r> getG2Filt();

    // std::tuple<stdvec, stdvec, stdvec, stdvec, stdvec> getPSD();

    stdvec_c getInterference();

    std::vector<stdvec_c> getAverageData();

    void setSubtractionTrace(std::vector<stdvec_c> trace);

    std::vector<stdvec_c> getSubtractionTrace();

    int getTotalLength() { return processor->getTotalLength(); };

    int getTraceLength() { return processor->getTraceLength(); };

    int getOutSize() { return processor->getOutSize(); };

    size_t getNotifySize() { return notify_size; };

    // std::vector<std::vector<std::complex<float>>> getFirwins()
    // {
    //     auto firwins = processor->getAllFirwins();
    //     std::vector<std::vector<std::complex<float>>> result(4);
    //     for (int i = 0; i < 4; i++)
    //         result[i] = std::vector<std::complex<float>>(firwins[i].begin(), firwins[i].end());
    //     return result;
    // };

    std::vector<stdvec> getRealResults();

    std::vector<stdvec_c> getComplexResults();

protected:
    void initializeBuffer();

    template <typename T, typename V>
    std::vector<V> postprocess(const thrust::host_vector<T> &data);

    template <template <typename, typename...> class Container, typename T, typename... Args>
    thrust::host_vector<T> tile(const Container<T, Args...> &data, size_t N);
};
#endif // MEASUREMENT_H