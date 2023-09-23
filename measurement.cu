//
// Created by andrei on 4/13/21.
//
#include <chrono>
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "yokogawa_gs210.h"
#include <future>
#include <thread>

namespace py = pybind11;

Measurement::Measurement(Digitizer *dig_, uint64_t averages, uint64_t batch, double part, int K, const char* coil_address)
{
    dig = dig_;
    sampling_rate = dig->getSamplingRate();
    coil = new yokogawa_gs210(coil_address);
    segment_size = dig->getSegmentSize();
    batch_size = batch;
    this->setAveragesNumber(averages);
    notify_size = 2 * segment_size * batch_size;
    dig->handleError();
    dig->setTimeout(5000);  // ms
    processor = new dsp(segment_size, batch_size, part, K, sampling_rate);
    this->initializeBuffer();

    func = [this](int8_t* data) mutable { this->processor->compute(data); };

    int trace_length = processor->getTraceLength();

    test_input.resize(notify_size * 2);
}

Measurement::Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch, double part, int K, const char* coil_address)
    : Measurement(new Digitizer(reinterpret_cast<void*>(dig_handle)), averages, batch, part, K, coil_address)
{}

void Measurement::initializeBuffer()
{
    // Create the buffer in page-locked memory
    size_t buffersize = 4 * notify_size;
    processor->createBuffer(buffersize);
    dig->setBuffer(processor->getBuffer(), buffersize);
}

void Measurement::setCurrents(float wc, float oc)
{
    working_current = wc; 
    offset_current = oc;
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
}

void Measurement::setAveragesNumber(uint64_t averages)
{
    segments_count = averages;
    iters_num = averages / batch_size;
    iters_done = 0;
}

void Measurement::setTapers(std::vector<stdvec> tapers)
{
    processor->setTapers(tapers);
}

void Measurement::setCalibration(float r, float phi, float offset_i, float offset_q)
{
    processor->setDownConversionCalibrationParameters(r, phi, offset_i, offset_q);
}

void Measurement::setFirwin(float left_cutoff, float right_cutoff)
{
    int oversampling = (int) std::round(1.25E+9f / dig->getSamplingRate());
    processor->setFirwin(left_cutoff, right_cutoff, oversampling);
}

void Measurement::measure()
{
    dig->prepareFifo(notify_size);
    dig->launchFifo(notify_size, iters_num, func, true);
    dig->stopFifo();
    iters_done += iters_num;
}

void Measurement::asyncCurrentSwitch()
{
    coil->set_current(working_current);
    setSubtractionTrace(getSubtractionData(), getSubtractionNoise());
    resetOutput();
}

void Measurement::measureWithCoil()
{
    coil->set_current(offset_current);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    dig->prepareFifo(notify_size);
    dig->launchFifo(notify_size, iters_num, func, true);
    iters_done += iters_num;

    uint64_t iters_delay = 1 * sampling_rate / notify_size * 2;
    //auto a = std::async(std::launch::async, &Measurement::asyncCurrentSwitch, this);
    //dig->launchFifo(notify_size, iters_delay, func, false);
    //a.wait();

    std::thread t1 (&Measurement::asyncCurrentSwitch, this);
    //std::thread t2 (&Digitizer::launchFifo, dig, notify_size, iters_delay, func, false);
    dig->launchFifo(notify_size, iters_delay, func, false);
    t1.join();
    //t2.join();
    //asyncCurrentSwitch();

    dig->launchFifo(notify_size, iters_num, func, true);
    iters_done += iters_num;
    dig->stopFifo();
}

void Measurement::measureTest()
{
    for (uint32_t i = 0; i < iters_num; i++)
        func(test_input.data());
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

    int8_t* input_ptr = (int8_t*)info.ptr;
    size_t len = 2 * segment_size;
    for (size_t i = 0; i < batch_size; i++)
        test_input.insert(test_input.begin() + i * len, input_ptr, input_ptr + len);
}

stdvec_c Measurement::getMeanField()
{
    auto field_form_gpu = processor->getCumulativeField();
    return postprocess(field_form_gpu);
}

stdvec Measurement::getMeanPower()
{
    auto power_form_gpu = processor->getCumulativePower();
    return postprocess(power_form_gpu);
}

stdvec Measurement::postprocess(hostvec& data)
{
    using namespace thrust::placeholders;
    stdvec result(data.size());
    float divider = (iters_done > 0) ? iters_done : 1;
    thrust::transform(data.begin(), data.end(), result.begin(), _1 / divider);
    return result;
}

stdvec_c Measurement::postprocess(hostvec_c& data)
{
    using namespace thrust::placeholders;
    stdvec_c result(data.size());
    float divider = (iters_done > 0) ? iters_done : 1;
    thrust::transform(data.begin(), data.end(), result.begin(), _1 / divider);
    return result;
}

stdvec Measurement::getPSD()
{
    auto psd_spectrum = processor->getPowerSpectrum();
    return postprocess(psd_spectrum);
}

stdvec_c Measurement::getDataSpectrum()
{
    auto data_spectrum = processor->getDataSpectrum();
    return postprocess(data_spectrum);
}

stdvec_c Measurement::getNoiseSpectrum()
{
    auto noise_spectrum = processor->getNoiseSpectrum();
    return postprocess(noise_spectrum);
}

stdvec Measurement::getPeriodogram()
{
    auto periodogram = processor->getPeriodogram();
    return postprocess(periodogram);
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

template <template <typename, typename...> class Container, typename T, typename... Args>
thrust::host_vector<T> Measurement::tile(const Container<T, Args...>& data, size_t N)
{
    // data : vector to tile
    // N : how much to tile
    using iter_t = typename Container<T, Args...>::const_iterator;
    thrust::host_vector<T> tiled_data;
    tiled_data.resize(data.size() * N);
    for (int i = 0; i < N; i++)
        tiled_data.insert(tiled_data.end(), data.begin(), data.end());
    return tiled_data;
}

void Measurement::setSubtractionTrace(stdvec_c trace, stdvec_c offsets)
{
    hostvec_c average = tile(trace, batch_size);
    hostvec_c average_offsets = tile(offsets, batch_size);
    processor->setSubtractionTrace(average, average_offsets);
}

stdvec_c Measurement::getSubtractionData()
{
    auto subtr_data_from_gpu = processor->getCumulativeSubtrData();
    return postprocess(subtr_data_from_gpu);
}

stdvec_c Measurement::getSubtractionNoise()
{
    auto subtr_noise_from_gpu = processor->getCumulativeSubtrNoise();
    return postprocess(subtr_noise_from_gpu);
}

py::tuple Measurement::getSubtractionTrace()
{
    int len = processor->getTotalLength();
    hostvec_c subtraction_trace(len);
    hostvec_c subtraction_offs(len);
    processor->getSubtractionTrace(subtraction_trace, subtraction_offs);
    return py::make_tuple(stdvec_c(subtraction_trace.begin(), subtraction_trace.end()), 
        stdvec_c(subtraction_offs.begin(), subtraction_offs.end()));
}


std::vector<std::vector<float>> Measurement::getDPSSTapers()
{
    auto tapers = processor->getDPSSTapers();

    size_t num_rows = tapers.size();
    size_t num_cols = (num_rows > 0) ? tapers[0].size() : 0;

    std::vector<std::vector<float>> result(num_rows);
    for (size_t i = 0; i < num_rows; ++i)
    {
        result[i].resize(num_cols);
        std::copy(tapers[i].begin(), tapers[i].end(), result[i].begin());
    }

    return result;
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
}

Measurement::~Measurement()
{
    if ((processor != NULL) || (dig != NULL))
        this->free();
}