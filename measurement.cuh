//
// Created by andrei on 4/13/21.
//
#ifndef SPECTRUMEXTENSION_MEASUREMENT_H
#define SPECTRUMEXTENSION_MEASUREMENT_H

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include "digitizer.h"
#include "dsp.cuh"
#include "pinned_allocator.cuh"
#include <pybind11/pybind11.h>
#include "yokogawa_gs210.h"

using corr_t = std::vector<std::vector<std::complex<double>>>;
using trace_t = std::vector<std::complex<double>>;

class Measurement
{
private:
    Digitizer *dig;
    yokogawa_gs210 *coil;
    dsp *processor;
    size_t segment_size;
    uint64_t segments_count;
    uint64_t batch_size;
    size_t notify_size;
    uint64_t iters_num;
    uint64_t iters_done;
    double sampling_rate;

    float offset_current = 0.f;
    float working_current = 0.f;

    float max = 0.f;

    thrust::host_vector<int8_t> test_input;

    proc_t func;
    proc_t func_ult_calib;

public:
    Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch, double part,
                int second_oversampling, int K, const char *coil_address);

    Measurement(Digitizer *dig_, uint64_t averages, uint64_t batch, double part,
                int second_oversampling, int K, const char *coil_address);

    void setAmplitude(int ampl);

    void setCurrents(float wc, float oc);

    void setAveragesNumber(uint64_t averages);

    ~Measurement();

    void reset();

    void resetOutput();

    void free();

    void setCalibration(float r, float phi, float offset_i, float offset_q);

    void setFirwin(float left_cutoff, float right_cutoff);

    void setTapers(std::vector<stdvec> tapers);

    void setIntermediateFrequency(float frequency);

    void measure();

    void asyncCurrentSwitch();

    void measureWithCoil();

    void measureTest();

    void setTestInput(const std::vector<int8_t> &input);

    stdvec_c getMeanField();

    stdvec getMeanPower();

    stdvec getPSD();

    stdvec_c getDataSpectrum();

    stdvec_c getNoiseSpectrum();

    stdvec getPeriodogram();

    void setSubtractionTraces(stdvec_c trace, stdvec_c offsets);

    std::pair<stdvec_c, stdvec_c> getSubtractionTraces();

    std::pair<stdvec_c, stdvec_c> getAccumulatedSubstractionData();

    int getTotalLength() { return processor->getTotalLength(); }

    int getTraceLength() { return processor->getTraceLength(); }

    size_t getNotifySize() { return notify_size; }

    std::vector<std::vector<float>> getDPSSTapers();

protected:
    void initializeBuffer();

    template <typename T, typename V>
    std::vector<V> postprocess(const thrust::host_vector<T> &data);

    template <template <typename, typename...> class Container, typename T, typename... Args>
    thrust::host_vector<T> tile(const Container<T, Args...> &data, size_t N);
};

#endif // SPECTRUMEXTENSION_MEASUREMENT_H
