//
// Created by andrei on 4/13/21.
//
#include <vector>
#include <memory>
#include <cstdint>
#include "digitizer.h"
#include "dsp.cuh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifndef SPECTRUMEXTENSION_MEASUREMENT_H
#define SPECTRUMEXTENSION_MEASUREMENT_H

class Measurement {
private:
    Digitizer* dig;
    dsp* processor;
    size_t segment_size;
    uint64_t segments_count;
    uint64_t batch_size;
    size_t notify_size;
    uint64_t iters_num;
    uint64_t iters_done;

    float max = 0.f;

    hostbuf test_input;

    proc_t func;
    proc_t func_ult_calib;

public:
    Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch, double part);

    Measurement(Digitizer *dig, uint64_t averages, uint64_t batch, double part);

    void setAmplitude(int ampl);

    void setAveragesNumber(uint64_t averages);

    ~Measurement();

    void reset();

    void resetOutput();

    void free();

    void setCalibration(float r, float phi, float offset_i, float offset_q);

    void setFirwin(float left_cutoff, float right_cutoff);
    
    void setIntermediateFrequency(float frequency);

    int getCounter();

    void measure();

    void measureTest();

    void setTestInput(py::buffer input);

    std::vector<std::complex<double>> getMeanField();

    std::vector<double> getMeanPower();

    std::vector<double> getMeanSpectrum();

    std::vector <std::vector<std::complex<double>>> getCorrelator();
    
    std::vector<std::complex<float>> getRawCorrelator();

    void setSubtractionTrace(std::vector<std::complex<float>> trace);

    std::vector<std::complex<float>> getSubtractionTrace();

    int getTotalLength() { return processor->getTotalLength(); }

    int getTraceLength() { return processor->getTraceLength(); }

    int getOutSize() { return processor->getOutSize(); }

    size_t getNotifySize() { return notify_size; }

protected:
    void initializeBuffer();
};

#endif //SPECTRUMEXTENSION_MEASUREMENT_H
