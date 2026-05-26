//
// Created by andrei on 4/13/21.
//
#ifndef SPECTRUMEXTENSION_MEASUREMENT_H
#define SPECTRUMEXTENSION_MEASUREMENT_H

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <tuple>
#include "digitizer.h"
#include "dsp.cuh"
#include "pinned_allocator.cuh"

// namespace py = pybind11;

using output_complex_t = std::complex<float>;
using trace_t = std::vector<output_complex_t>;
using corr_t = std::vector<trace_t>;


class Measurement
{
private:
    std::unique_ptr<Digitizer> dig;
    std::unique_ptr<dsp> processor;
    size_t segment_size = 0;
    uint64_t segments_count = 0;
    uint64_t batch_size = 0;
    size_t notify_size = 0;
    uint64_t iters_num = 0;
    uint64_t iters_done = 0;
    double sampling_rate = 0.0;

    int second_ovs = 1;

    thrust::host_vector<int8_t> test_input;

    proc_t func;

    dsp &requireProcessor();
    const dsp &requireProcessor() const;
    Digitizer &requireDigitizer();
    Measurement(std::unique_ptr<Digitizer> dig_, uint64_t averages, uint64_t batch,
                int second_oversampling, const std::string &result_mode);

public:
    Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch,
                int second_oversampling, const std::string &result_mode = "average_g1");

    Measurement(Digitizer *dig_, uint64_t averages, uint64_t batch,
                int second_oversampling, const std::string &result_mode = "average_g1");

    Measurement(uint64_t averages, uint64_t batch, long segment, int dig_oversampling,
                int second_oversampling, const std::string &result_mode = "average_g1");
    
    void setDigParameters();
                
    void setAmplitude(int ampl);

    void setAveragesNumber(uint64_t averages);

    ~Measurement();

    void reset();

    void resetOutput();

    void free();

    void setCalibration(int line_num, float r, float phi, float offset_i, float offset_q);

    void setFirwin(float left_cutoff, float right_cutoff);
    void setFirwin(const stdvec_c window);

    void setIntermediateFrequency(float frequency);

    void setCorrDowncovertCoeffs(float freq1, float freq2);

    void measure();

    void measureTest();

    void setTestInput(const std::vector<int8_t> &input);

    corr_t getG1Correlator();

    std::tuple<corr_t, corr_t, corr_t> getG1OtherCorrelators();

    std::pair<stdvec_c, stdvec_c> getAverageField();

    std::pair<std::complex<float>, std::complex<float>> getS21();

    stdvec_c getCrossPower();

    stdvec_c getCrossSpectrum();

    std::vector<stdvec_c> getSubtractionData();

    void setSubtractionTrace(std::vector<stdvec_c> trace);

    std::vector<stdvec_c> getSubtractionTrace();

    int getTotalLength() { return requireProcessor().getTotalLength(); }

    int getTraceLength() { return requireProcessor().getTraceLength(); }

    int getResampledTraceLength() { return requireProcessor().getResampledTraceLength(); }

    int getOutSize() { return requireProcessor().getOutSize(); }

    size_t getNotifySize() { requireProcessor(); return notify_size; }

    std::string getResultMode() const { return requireProcessor().getResultModeName(); }

protected:
    void initializeBuffer();

    float getIterationsDivisor() const;

    corr_t makeCorrelationMatrix(const hostvec_c &data, int side) const;

    template <typename T, typename V>
    std::vector<V> postprocess(const thrust::host_vector<T> &data);

    template <template <typename, typename...> class Container, typename T, typename... Args>
    thrust::host_vector<T> tile(const Container<T, Args...> &data, size_t N);
};

#endif // SPECTRUMEXTENSION_MEASUREMENT_H
