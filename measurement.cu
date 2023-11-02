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
#include "dsp_functors.cuh"
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

void Measurement::reset()
{
    resetOutput();
    processor->resetSubtractionTrace();
}

void Measurement::resetOutput()
{
    iters_done = 0;
    processor->resetOutput();
}
void Measurement::initializeBuffer()
{
    // Create the buffer in page-locked memory
    size_t buffersize = 4 * notify_size;
    buffer.resize(buffersize, 0);
    dig->setBuffer(buffer.data(), buffersize);
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
    int oversampling = static_cast<int>(std::round(1.25E+9 / sampling_rate));
    processor->setIntermediateFrequency(frequency, oversampling);
    cudaDeviceSynchronize();
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
    int oversampling = static_cast<int>(std::round(1.25E+9f / dig->getSamplingRate()));
    processor->setFirwin(left_cutoff, right_cutoff, oversampling);
    cudaDeviceSynchronize();
}

void Measurement::setCorrelationFirwin(float cutoff_1[2], float cutoff_2[2])
{
    int oversampling = static_cast<int>(std::round(1.25E+9f / dig->getSamplingRate()));
    processor->setCorrelationFirwin(cutoff_1, cutoff_2, oversampling);
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
    setSubtractionTrace(getSubtractionData(), getSubtractionNoise());
    resetOutput();
    cudaDeviceSynchronize();
}

void Measurement::measureWithCoil()
{
    coil->set_current(offset_current);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
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

std::vector<std::vector<std::complex<double>>> Measurement::getCorrelator(string request)
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    hostvec_c result(len);
    std::vector<std::vector<std::complex<double>>> average_result(
        side, std::vector<std::complex<double>>(side));

    // Receive data from GPU
    if (request == "g1")
    {
        processor->getG1results(result);
    }
    elif (request == "g2_full")
    {
        processor->getG2FullResults(result);
    }
    elif (request == "g2_filteted")
    {
        processor->getG2FilteredResults(result);
    }
    else
    {
        std::cerr << "Request is not correct";
        return 1;
    }
    
    // Divide the data by a number of traces measured
    int k = 0;
    tcf X((iters_done > 0) ? static_cast<float>(iters_done * batch_size) : 1.f, 0.f);
  
    for (int t1 = 0; t1 < side; t1++)
        for (int t2 = t1; t2 < side; t2++)
        {
            average_result[t1][t2] = std::complex<float>(corr[t1 * side + t2]);
            average_result[t2][t1] = std::conj(average_result[t1][t2]);
        }
        
    return average_result;
}

stdvec_c Measurement::getRawG1()
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    hostvec_c result(len);

    // Receive data from GPU
    processor->getG1results(result);

    return stdvec_c(result.begin(), result.end());
}

stdvec_c Measurement::getRawG2()
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    hostvec_c result(len);

    // Receive data from GPU
    processor->getG2FullResults(result);

    return stdvec_c(result.begin(), result.end());
}

void Measurement::setSubtractionTrace(stdvec_c trace, stdvec_c offsets)
{
    hostvec_c average = tile(trace, batch_size);
    hostvec_c average_offsets = tile(offsets, batch_size);
    processor->setSubtractionTrace(average, average_offsets);
}

stdvec_c Measurement::getSubtractionData()
{
    return postprocess<tcf, std::complex<float>>(processor->getCumulativeSubtrData());
}

stdvec_c Measurement::getSubtractionNoise()
{
    return postprocess<tcf, std::complex<float>>(processor->getCumulativeSubtrNoise());
}

py::tuple Measurement::getSubtractionTrace()
{
    auto len = processor->getTotalLength();
    hostvec_c subtraction_trace(len);
    hostvec_c subtraction_offs(len);
    processor->getSubtractionTrace(subtraction_trace, subtraction_offs);
    return py::make_tuple(stdvec_c(subtraction_trace.begin(), subtraction_trace.end()),
                          stdvec_c(subtraction_offs.begin(), subtraction_offs.end()));
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