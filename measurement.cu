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
#include "yokogawa_gs210.h"
#include <thrust/zip_function.h>
#include <future>
#include <thread>


Measurement::Measurement(Digitizer *dig_, uint64_t averages, uint64_t batch, double part,
                         int second_oversampling, const char *coil_address)
{
    dig = dig_;
    sampling_rate = static_cast<double>(dig->getSamplingRate());
    coil = new yokogawa_gs210(coil_address);
    segment_size = dig->getSegmentSize();
    batch_size = batch;
    setAveragesNumber(averages);
    notify_size = 2 * num_channels * segment_size * batch_size;
    dig->handleError();
    dig->setTimeout(5000); // ms
    processor = new dsp(segment_size, batch_size, part, sampling_rate, second_oversampling);
    initializeBuffer();

    func = [this](int8_t *data) mutable
    { processor->compute(data); };

    int trace_length = processor->getTraceLength();

    test_input.resize(notify_size * 2, 0);
}

Measurement::Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch, double part,
                         int second_oversampling, const char *coil_address)
    : Measurement(new Digitizer(reinterpret_cast<void *>(dig_handle)), averages, batch, part,
                  second_oversampling, coil_address)
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
    processor->createBuffer(buffersize);
    // size_t interbuffersize = processor->getInterBufferSize();
    processor->createInterBuffer(buffersize / num_channels);
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
    setSubtractionTrace(getSubtractionData());
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


std::vector<std::vector<std::complex<double>>> Measurement::getCorrelator(std::string request)
{
    int len = processor->getOutSize();
    int side = processor->getTraceLength();

    hostvec_c result(len);
    std::vector<std::vector<std::complex<double>>> average_result(
        side, std::vector<std::complex<double>>(side));

    // Receive data from GPU
    if (request == "g2_full")
        processor->getG2FullResults(result);
    else if (request == "g2_filtered")
        processor->getG2FilteredResults(result);
    else
        throw std::runtime_error("Request is not correct");
    
    // Divide the data by a number of traces measured
    tcf X((iters_done > 0) ? static_cast<float>(iters_done * batch_size) : 1.f, 0.f);
  
    for (int t1 = 0; t1 < side; t1++)
        for (int t2 = t1; t2 < side; t2++)
        {
            average_result[t1][t2] = std::complex<float>(result[t1 * side + t2]);
            average_result[t2][t1] = std::conj(average_result[t1][t2]);
        }
        
    return average_result;
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

void Measurement::setSubtractionTrace(std::vector<stdvec_c> trace)
{
    hostvec_c average[num_channels];
    for (int i = 0; i < num_channels; i++)
    {
        average[i] = tile(trace[i], batch_size);
    }
    processor->setSubtractionTrace(average);
}

std::vector<stdvec_c> Measurement::getSubtractionData()
{
    std::vector<stdvec_c> subtr_data;
    auto vec = processor->getCumulativeSubtrData();
    for (int i = 0; i < num_channels; i++)
    {
        subtr_data.push_back(postprocess<tcf, std::complex<float>>(vec[i]));
    }
    
    return subtr_data;
}

std::vector<hostvec_c> Measurement::getSubtractionTrace()
{
    auto len = processor->getTotalLength();
    std::vector<hostvec_c> subtraction_trace;
    processor->getSubtractionTrace(subtraction_trace);
    return subtraction_trace;
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