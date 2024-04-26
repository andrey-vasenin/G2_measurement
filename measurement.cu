//
// Created by andrei on 4/13/21.
//
#include <chrono>
#include <memory>
#include <iostream>
#include <functional>
#include <vector>
#include <span>
#include <numeric>
#include <complex>
#include <cstdint>
#include "dsp.cuh"
#include "digitizer.h"
#include "measurement.cuh"
#include "tiled_range.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "yokogawa_gs210.h"
#include <thrust/zip_function.h>
#include <future>
#include <thread>

namespace py = pybind11;

Measurement::Measurement(Digitizer *dig_, uint64_t averages, uint64_t batch, double part,
                         int second_oversampling, int K, const char *coil_address)
{
    dig = dig_;
    sampling_rate = static_cast<double>(dig->getSamplingRate());
    coil = new yokogawa_gs210(coil_address);
    segment_size = dig->getSegmentSize();
    batch_size = batch;
    setAveragesNumber(averages);
    notify_size = 2 * segment_size * batch_size;
    dig->handleError();
    dig->setTimeout(5000); // ms
    processor = new dsp(segment_size, batch_size, part, K, sampling_rate, second_oversampling);
    initializeBuffer();

    func = [this](int8_t *data) mutable
    { processor->compute(data); };

    int trace_length = processor->getTraceLength();

    test_input.resize(notify_size * 2, 0);
}

Measurement::Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch, double part,
                         int second_oversampling, int K, const char *coil_address)
    : Measurement(new Digitizer(reinterpret_cast<void *>(dig_handle)), averages, batch, part,
                  second_oversampling, K, coil_address)
{
}
// Constructor for test measurement
Measurement::Measurement(uint64_t averages, uint64_t batch, long segment, double part,
                int second_oversampling, int K)
{
    dig = nullptr;
    batch_size = batch;
    segment_size = segment;
    sampling_rate = 1.25E+9;
    setAveragesNumber(averages);
    notify_size = 2 * segment_size * batch_size;
    processor = new dsp(segment_size, batch_size, part, K, sampling_rate, second_oversampling);
    initializeBuffer();

    func = [this](int8_t *data) mutable
    { processor->compute(data); };
    int trace_length = processor->getTraceLength();

    test_input.resize(notify_size, 0);
}

Measurement::~Measurement()
{
    if ((processor != nullptr) || (dig != nullptr))
        free();
}

void Measurement::free()
{
    delete processor;
    delete dig;
    processor = nullptr;
    dig = nullptr;
}

void Measurement::resetOutput()
{
    iters_done = 0;
    processor->resetOutput();
}

void Measurement::reset()
{
    resetOutput();
    processor->resetSubtractionTrace();
}

void Measurement::initializeBuffer()
{
    // Create the buffer in page-locked memory
    size_t buffersize = 4 * notify_size; // buffersize should be a multiple of 4 kByte
    processor->createBuffer(buffersize);
    if (dig != nullptr)
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

void Measurement::setWelchWindow()
{
    processor->setWelchWindow();
}

/* Use frequency in GHz */
void Measurement::setIntermediateFrequency(float frequency)
{
    int oversampling = static_cast<int>(std::round(1.25E+9 / sampling_rate));
    processor->setIntermediateFrequency(frequency, oversampling);
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
    long sr = 0;
    if (dig != nullptr)
        sr = dig->getSamplingRate();
    else 
        sr = sampling_rate;
    int oversampling = static_cast<int>(std::round(1.25E+9f / sr));
    processor->setFirwin(left_cutoff, right_cutoff, oversampling);
    cudaDeviceSynchronize();
}

void Measurement::measure()
{
    dig->prepareFifo(static_cast<unsigned long>(notify_size));
    dig->launchFifo(static_cast<unsigned long>(notify_size), iters_num, func, true);
    dig->stopFifo();
    iters_done += iters_num;
}

void Measurement::asyncCurrentSwitch()
{
    coil->set_current(working_current);
    auto subtr_traces = getAccumulatedSubstractionData();
    resetOutput();
    setSubtractionTraces(std::get<0>(subtr_traces), std::get<1>(subtr_traces));
}

void Measurement::measureWithCoil()
{
    coil->set_current(offset_current);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    dig->prepareFifo(notify_size);
    dig->launchFifo(notify_size, iters_num, func, true);
    iters_done += iters_num;

    uint64_t iters_delay = static_cast<size_t>(sampling_rate) / notify_size * 2;
    // auto a = std::async(std::launch::async, &Measurement::asyncCurrentSwitch, this);
    // dig->launchFifo(notify_size, iters_delay, func, false);
    // a.wait();

    std::thread t1(&Measurement::asyncCurrentSwitch, this);
    // std::thread t2 (&Digitizer::launchFifo, dig, notify_size, iters_delay, func, false);
    dig->launchFifo(notify_size, iters_delay, func, false);
    t1.join();
    // t2.join();
    // asyncCurrentSwitch();

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

void Measurement::setTestInput(const std::vector<int8_t> &input)
{
    if (input.size() < 2 * segment_size)
        throw std::runtime_error("Number of element in the input array "
                                 "must be larger or equal to the two segment sizes");
    test_input = tile(input, batch_size);
}

stdvec_c Measurement::getMeanField()
{
    return postprocess<tcf, std::complex<float>>(processor->getCumulativeField());
}

stdvec Measurement::getMeanPower()
{
    return postprocess<float, float>(processor->getCumulativePower());
}

stdvec Measurement::getPSD()
{
    return postprocess<float, float>(processor->getPowerSpectrum());
}

stdvec_c Measurement::getDataSpectrum()
{
    return postprocess<tcf, std::complex<float>>(processor->getDataSpectrum());
}

stdvec_c Measurement::getNoiseSpectrum()
{
    return postprocess<tcf, std::complex<float>>(processor->getNoiseSpectrum());
}

stdvec Measurement::getPeriodogram()
{
    return postprocess<float, float>(processor->getPeriodogram());
}

void Measurement::setSubtractionTraces(stdvec_c trace, stdvec_c offsets)
{
    hostvec_c average = tile(trace, batch_size);
    hostvec_c average_offsets = tile(offsets, batch_size);
    processor->setSubtractionTraces(average, average_offsets);
}

std::pair<stdvec_c, stdvec_c> Measurement::getSubtractionTraces()
{
    auto [subtraction_data, subtraction_noise] = processor->getSubtractionTraces();
    return {stdvec_c(subtraction_data.begin(), subtraction_data.end()),
            stdvec_c(subtraction_noise.begin(), subtraction_noise.end())};
}

std::pair<stdvec_c, stdvec_c> Measurement::getAccumulatedSubstractionData()
{
    auto [rd, rn] = processor->getCumulativeSubtrData();
    auto xd = postprocess<tcf, std::complex<float>>(rd);
    auto xn = postprocess<tcf, std::complex<float>>(rn);
    return {xd, xn};
}

stdvec Measurement::getWelchSpectrum()
{
    return postprocess<float, float>(processor->getWelchSpectrum());
}

void Measurement::setTapers(std::vector<stdvec> tapers)
{
    processor->setTapers(tapers);
}

std::vector<std::vector<float>> Measurement::getDPSSTapers()
{
    auto tapers = processor->getDPSSTapers();
    std::vector<std::vector<float>> result(tapers.size());
    thrust::for_each(thrust::make_zip_iterator(tapers.cbegin(), result.begin()),
                     thrust::make_zip_iterator(tapers.cend(), result.end()),
                     thrust::make_zip_function([](const auto &src, auto &dst)
                                               {
        dst.resize(src.size());
        std::copy(src.cbegin(), src.cend(), dst.begin()); }));
    return result;
}

template <typename T, typename V>
std::vector<V> Measurement::postprocess(const thrust::host_vector<T> &data)
{
    std::vector<V> result(data.size());
    float divider = (iters_done > 0) ? static_cast<float>(iters_done) : 1.f;
    thrust::transform(data.cbegin(), data.cend(), result.begin(),
                      [divider](const T &x)
                      { return static_cast<V>(x / divider); });
    return result;
}

template <template <typename, typename...> class Container, typename T, typename... Args>
thrust::host_vector<T> Measurement::tile(const Container<T, Args...> &data, size_t N)
{
    // data : vector to tile
    // N : how much to tile
    using iter_t = typename Container<T, Args...>::const_iterator;
    thrust::host_vector<T> tiled_data(data.size() * N);
    tiled_range<iter_t> tiled_iter(data.begin(), data.end(), N);
    thrust::copy(tiled_iter.begin(), tiled_iter.end(), tiled_data.begin());
    return tiled_data;
}