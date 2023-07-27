//
// Created by andrei on 4/13/21.
//
#include <memory>
#include <iostream>
#include <functional>
#include <vector>
#include <numeric>
#include <complex>
#include <cstdint>
#include "dsp.cuh"
#include "digitizer.h"
#include "measurement.cuh"
#include "tiled_range.cuh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

Measurement::Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch, double part)
{
    dig = new Digitizer(reinterpret_cast<void*>(dig_handle));
    segment_size = dig->getSegmentSize();
    batch_size = batch;
    this->setAveragesNumber(averages);
    notify_size = 2 * segment_size * batch_size;
    dig->handleError();
    dig->setTimeout(5000);  // ms
    processor = new dsp(segment_size, batch_size, part);
    this->initializeBuffer();

    func = [this](int8_t* data) mutable { this->processor->compute(data); };

    int trace_length = processor->getTraceLength();

    test_input.resize(notify_size * 2);
}

Measurement::Measurement(Digitizer *dig_, uint64_t averages, uint64_t batch, double part)
{
    dig = dig_;
    segment_size = dig->getSegmentSize();
    batch_size = batch;
    this->setAveragesNumber(averages);
    notify_size = 2 * segment_size * batch_size;
    dig->handleError();
    dig->setTimeout(5000);  // ms
    processor = new dsp(segment_size, batch_size, part);
    this->initializeBuffer();

    func = [this](int8_t* data) mutable { this->processor->compute(data); };

    int trace_length = processor->getTraceLength();

    test_input.resize(notify_size * 2);
}

void Measurement::initializeBuffer()
{
    // Create the buffer in page-locked memory
    size_t buffersize = 4 * notify_size;
    processor->createBuffer(buffersize);
    dig->setBuffer(processor->getBuffer(), buffersize);
}

void Measurement::setAmplitude(int ampl)
{
    processor->setAmplitude(ampl);
}

/* Use frequency in GHz */
void Measurement::setIntermediateFrequency(float frequency)
{
    int oversampling = (int)std::round(1.25E+9f / dig->getSamplingRate());
    processor->setIntermediateFrequency(frequency, oversampling);
    cudaDeviceSynchronize();
}

void Measurement::setAveragesNumber(uint64_t averages)
{
    segments_count = averages;
    iters_num = averages / batch_size;
    iters_done = 0;
}

void Measurement::setCalibration(float r, float phi, float offset_i, float offset_q)
{
    processor->setDownConversionCalibrationParameters(r, phi, offset_i, offset_q);
}

void Measurement::setFirwin(float left_cutoff, float right_cutoff)
{
    int oversampling = (int) std::round(1.25E+9f / dig->getSamplingRate());
    processor->setFirwin(left_cutoff, right_cutoff, oversampling);
    cudaDeviceSynchronize();
}

int Measurement::getCounter()
{
    return processor->getCounter();
}

void Measurement::measure()
{
    dig->launchFifo(notify_size, iters_num, func);
    iters_done += iters_num;
}

void Measurement::measureTest()
{
    for (uint32_t i = 0; i < iters_num; i++)
        func(&test_input[0]);
    iters_done += iters_num;
}

void Measurement::setTestInput(py::buffer input)
{
    py::buffer_info info = input.request();
    if (info.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    if (static_cast<size_t>(info.size) < 2 * segment_size)
        throw std::runtime_error("Number of element in the imput array "
            "must be larger or equal to the two segment sizes");

    char* input_ptr = (char*)info.ptr;
    tiled_range<char*> tiled_input(input_ptr, input_ptr + 2 * segment_size, batch_size);
    thrust::copy(tiled_input.begin(), tiled_input.end(), test_input.begin());
}

std::vector<std::complex<double>> Measurement::getMeanField()
{
    int len = processor->getTotalLength();
    int tl = processor->getTraceLength();

    hostvec_c field_from_gpu(len);
    processor->getCumulativeField(field_from_gpu);

    // Compute mean 
    std::vector<std::complex<double>> mean_field(tl, 0.);
    double denominator{ 1 };
    if (iters_done > 0)
        denominator = static_cast<double>(iters_done * batch_size);
    for (int j = 0; j < tl; j++)
    {
        for (int i = 0; i < batch_size; i++)
        {
            int idx = i * tl + j;
            std::complex<double> fval(field_from_gpu[idx]);
            mean_field[j] += fval;
        }
        mean_field[j] /= denominator;
    }
    return mean_field;
}

std::vector<double> Measurement::getMeanPower()
{
    int len = processor->getTotalLength();
    int tl = processor->getTraceLength();

    hostvec power_from_gpu(len);
    processor->getCumulativePower(power_from_gpu);

    // Compute mean 
    std::vector<double> mean_power(tl, 0.);
    double denominator{ 1 };
    if (iters_done > 0)
        denominator = static_cast<double>(iters_done * batch_size);
    for (int j = 0; j < tl; j++)
    {
        for (int i = 0; i < batch_size; i++)
        {
            int idx = i * tl + j;
            mean_power[j] += power_from_gpu[idx];
        }
        mean_power[j] /= denominator;
    }
    return mean_power;
}

std::vector<double> Measurement::getMeanSpectrum()
{
    int len = processor->getTotalLength();
    int tl = processor->getTraceLength();

    hostvec spec_from_gpu(len);
    processor->getCumulativeSpectrum(spec_from_gpu);

    // Compute mean 
    std::vector<double> mean_spec(tl, 0.);
    double denominator{ 1 };
    if (iters_done > 0)
        denominator = static_cast<double>(iters_done * batch_size);
    for (int j = 0; j < tl; j++)
    {
        for (int i = 0; i < batch_size; i++)
        {
            int idx = i * tl + j;
            mean_spec[j] += spec_from_gpu[idx];
        }
        mean_spec[j] /= denominator;
    }
    return mean_spec;
}


std::vector <std::vector<std::complex<double>>> Measurement::getCorrelator()
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    hostvec_c result(len);
    std::vector <std::vector<std::complex<double>>> average_result(
        side, std::vector<std::complex<double>>(side));

    // Receive data from GPU
    processor->getCorrelator(result);

    // Divide the data by a number of traces measured
    int k = 0;
    tcf X((iters_done > 0) ? static_cast<float>(iters_done * batch_size) : 1.f, 0.f);
    for (int t1 = 0; t1 < side; t1++)
    {
        for (int t2 = t1; t2 < side; t2++)
        {
            k = t1 * side + t2;
            average_result[t1][t2] = std::complex<double>(result[k] / X);
            average_result[t2][t1] = std::conj(average_result[t1][t2]);
        }
    }
    return average_result;
}

stdvec_c Measurement::getRawCorrelator()
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    hostvec_c result(len);

    // Receive data from GPU
    processor->getCorrelator(result);

    return stdvec_c(result.begin(), result.end());
}

void Measurement::setSubtractionTrace(stdvec_c trace)
{
    //using namespace std::complex_literals;
    int N = processor->getTraceLength();
    int M = processor->getTotalLength();
    
    hostvec_c average(M);
    tiled_range<stdvec_c::iterator> tiled_input(trace.begin(), trace.end(), batch_size);
    thrust::copy(tiled_input.begin(), tiled_input.end(), average.begin());

    processor->setSubtractionTrace(average);
}

stdvec_c Measurement::getSubtractionTrace()
{
    int len = processor->getTotalLength();
    hostvec_c subtraction_trace(len);
    processor->getSubtractionTrace(subtraction_trace);
    return stdvec_c(subtraction_trace.begin(), subtraction_trace.end());
}

void Measurement::reset()
{
    this->resetOutput();
    processor->resetSubtractionTrace();
}

void Measurement::resetOutput()
{
    iters_done = 0;
    processor->resetOutput();
}

void Measurement::free()
{
    delete processor;
    delete dig;
    processor = NULL;
    dig = NULL;
    test_input.clear();
}

Measurement::~Measurement()
{
    if ((processor != NULL) || (dig != NULL))
        this->free();
}