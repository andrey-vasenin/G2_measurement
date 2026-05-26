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
#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include "dsp.cuh"
#include "dsp_functors.cuh"
#include "digitizer.h"
#include "measurement.cuh"
#include "tiled_range.cuh"
#include <thrust/zip_function.h>
#include <future>
#include <thread>


namespace
{
ResultMode parseResultMode(std::string mode)
{
    std::transform(mode.begin(), mode.end(), mode.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::replace(mode.begin(), mode.end(), '-', '_');

    if (mode == "average" || mode == "average_only" || mode == "s21")
        return ResultMode::AverageOnly;
    if (mode == "average_g1" || mode == "regular" || mode == "pulse")
        return ResultMode::AverageG1;
    if (mode == "all" || mode == "all_correlators")
        return ResultMode::AllCorrelators;

    throw std::runtime_error("Unsupported result_mode '" + mode + "'. Supported modes: average, average_g1, all_correlators");
}

void validateSecondOversampling(int second_oversampling)
{
    if (second_oversampling != 1 && second_oversampling != 2 && second_oversampling != 4)
        throw std::runtime_error("second_oversampling must be 1, 2, or 4");
}

void validateAverages(uint64_t averages, uint64_t batch)
{
    if (averages == 0)
        throw std::runtime_error("averages must be > 0");
    if (batch == 0)
        throw std::runtime_error("batch must be > 0");
    if (averages % batch != 0)
        throw std::runtime_error("averages must be divisible by batch");
}

size_t validateSegment(long segment)
{
    if (segment <= 0)
        throw std::runtime_error("segment must be > 0");
    return static_cast<size_t>(segment);
}

size_t validateSegmentSize(size_t segment)
{
    if (segment == 0)
        throw std::runtime_error("segment must be > 0");
    return segment;
}

void validateSegmentOversampling(size_t segment, int second_oversampling)
{
    validateSecondOversampling(second_oversampling);
    if (segment % static_cast<size_t>(second_oversampling) != 0)
        throw std::runtime_error("segment must be divisible by second_oversampling");
}

void validateDigitizerOversampling(int dig_oversampling)
{
    if (dig_oversampling <= 0)
        throw std::runtime_error("digitizer_oversampling must be > 0");
}

void validateSamplingRate(double sampling_rate)
{
    if (sampling_rate <= 0.0)
        throw std::runtime_error("sampling_rate must be > 0");
}

void validateSizeEquals(size_t actual, size_t expected, const char *name)
{
    if (actual != expected)
        throw std::runtime_error(std::string(name) + " must contain exactly " + std::to_string(expected) + " elements");
}

size_t checkedNotifySize(size_t segment, uint64_t batch)
{
    const size_t bytes_per_complex_pair = 2 * num_channels;
    if (batch > std::numeric_limits<size_t>::max())
        throw std::runtime_error("batch is too large");
    size_t batch_size = static_cast<size_t>(batch);
    if (segment != 0 && batch_size > std::numeric_limits<size_t>::max() / segment)
        throw std::runtime_error("segment * batch is too large");
    size_t samples = segment * batch_size;
    if (samples > std::numeric_limits<size_t>::max() / bytes_per_complex_pair)
        throw std::runtime_error("notify_size is too large");
    return bytes_per_complex_pair * samples;
}

std::unique_ptr<Digitizer> makeDigitizerFromHandle(std::uintptr_t dig_handle)
{
    if (dig_handle == 0)
        throw std::runtime_error("digitizer_handle must be nonzero");
    return std::make_unique<Digitizer>(reinterpret_cast<void *>(dig_handle));
}
}



Measurement::Measurement(std::unique_ptr<Digitizer> dig_guard, uint64_t averages, uint64_t batch,
                         int second_oversampling, const std::string &result_mode)
{
    if (dig_guard == nullptr)
        throw std::runtime_error("digitizer must not be null");

    ResultMode parsed_mode = parseResultMode(result_mode);
    segment_size = dig_guard->getSegmentSize();
    dig_guard->handleError();
    sampling_rate = static_cast<double>(dig_guard->getSamplingRate());
    dig_guard->handleError();
    segment_size = validateSegmentSize(segment_size);
    validateSamplingRate(sampling_rate);
    validateSegmentOversampling(segment_size, second_oversampling);
    validateAverages(averages, batch);

    batch_size = batch;
    second_ovs = second_oversampling;
    notify_size = checkedNotifySize(segment_size, batch_size);
    segments_count = averages;
    iters_num = averages / batch_size;
    iters_done = 0;
    dig_guard->setTimeout(5000); // ms
    dig_guard->handleError();

    std::unique_ptr<dsp> processor_guard(new dsp(segment_size, batch_size, sampling_rate, second_oversampling, parsed_mode));
    processor = std::move(processor_guard);
    dig = std::move(dig_guard);
    initializeBuffer();

    func = [this](int8_t *data) mutable
    {
        dsp &active_processor = requireProcessor();
        int stream_num = active_processor.compute(data);
        active_processor.waitInputCopy(stream_num);
    };

    test_input.resize(notify_size * 2, 0);
}

Measurement::Measurement(Digitizer *dig_, uint64_t averages, uint64_t batch,
                         int second_oversampling, const std::string &result_mode)
    : Measurement(std::unique_ptr<Digitizer>(dig_), averages, batch, second_oversampling, result_mode)
{
}

void Measurement::setDigParameters()
{
    Digitizer &active_digitizer = requireDigitizer();
    int channels[] = {0, 1, 2, 3};
    int amps[] = {1000, 1000, 1000, 1000};

    active_digitizer.setupChannels(channels, amps, 4);

    active_digitizer.setSamplingRate(1250000000 / 4);
    active_digitizer.setupSingleRecFifoMode(32);
    active_digitizer.setSegmentSize(800);

}

Measurement::Measurement(std::uintptr_t dig_handle, uint64_t averages, uint64_t batch,
                         int second_oversampling, const std::string &result_mode)
    : Measurement(makeDigitizerFromHandle(dig_handle), averages, batch,
                  second_oversampling, result_mode)
{
}

// Constructor for test measurement
Measurement::Measurement(uint64_t averages, uint64_t batch, long segment, int dig_oversampling,
                int second_oversampling, const std::string &result_mode)
{
    ResultMode parsed_mode = parseResultMode(result_mode);
    validateDigitizerOversampling(dig_oversampling);
    segment_size = validateSegment(segment);
    validateSegmentOversampling(segment_size, second_oversampling);
    validateAverages(averages, batch);

    batch_size = batch;
    second_ovs = second_oversampling;
    sampling_rate = 1.25E+9/dig_oversampling;
    validateSamplingRate(sampling_rate);
    notify_size = checkedNotifySize(segment_size, batch_size);
    segments_count = averages;
    iters_num = averages / batch_size;
    iters_done = 0;

    std::unique_ptr<dsp> processor_guard(new dsp(segment_size, batch_size, sampling_rate, second_oversampling, parsed_mode));
    processor = std::move(processor_guard);
    initializeBuffer();

    func = [this](int8_t *data) mutable
    {
        dsp &active_processor = requireProcessor();
        int stream_num = active_processor.compute(data);
        active_processor.waitInputCopy(stream_num);
    };
    test_input.resize(notify_size, 0);
}

Measurement::~Measurement() = default;

void Measurement::free()
{
    processor.reset();
    dig.reset();
    func = nullptr;
}

dsp &Measurement::requireProcessor()
{
    if (processor == nullptr)
        throw std::runtime_error("AverageFieldMeasurer has been freed or was not initialized");
    return *processor;
}

const dsp &Measurement::requireProcessor() const
{
    if (processor == nullptr)
        throw std::runtime_error("AverageFieldMeasurer has been freed or was not initialized");
    return *processor;
}

Digitizer &Measurement::requireDigitizer()
{
    if (dig == nullptr)
        throw std::runtime_error("measure requires a digitizer handle");
    return *dig;
}

void Measurement::reset()
{
    resetOutput();
    requireProcessor().resetSubtractionTrace();
}

void Measurement::resetOutput()
{
    iters_done = 0;
    requireProcessor().resetOutput();
}

void Measurement::initializeBuffer()
{
    // Create the buffer in page-locked memory
    size_t buffersize = 4 * notify_size; // buffersize should be a multiple of 4 kByte
    dsp &active_processor = requireProcessor();
    active_processor.createBuffer(buffersize);
    if (dig != nullptr)
        dig->setBuffer(active_processor.getBuffer(), buffersize);
}


void Measurement::setAmplitude(int ampl)
{
    requireProcessor().setAmplitude(ampl);
}

/* Use frequency in GHz */
void Measurement::setIntermediateFrequency(float frequency)
{
    int oversampling = static_cast<int>(std::round(1.25E+9 / sampling_rate));
    requireProcessor().setIntermediateFrequency(frequency, oversampling);
    cudaDeviceSynchronize();
}

void Measurement::setCorrDowncovertCoeffs(float freq1, float freq2)
{
    int oversampling = static_cast<int>(std::round(1.25E+9 / sampling_rate));
    requireProcessor().setCorrDowncovertCoeffs(freq1, freq2, oversampling * second_ovs);
}

void Measurement::setAveragesNumber(uint64_t averages)
{
    requireProcessor();
    validateAverages(averages, batch_size);
    segments_count = averages;
    iters_num = averages / batch_size;
    iters_done = 0;
}

void Measurement::setCalibration(int line_num, float r, float phi, float offset_i, float offset_q)
{
    if (line_num < 0 || line_num >= num_channels)
        throw std::runtime_error("line_num must be 0 or 1");
    requireProcessor().setDownConversionCalibrationParameters(line_num, r, phi, offset_i, offset_q);
}

void Measurement::setFirwin(float left_cutoff, float right_cutoff)
{   
    long sr = 0;
    if (dig != nullptr)
        sr = dig->getSamplingRate();
    else 
        sr = sampling_rate;
    int oversampling = static_cast<int>(std::round(1.25E+9f / sr));
    requireProcessor().setFirwin(left_cutoff, right_cutoff, oversampling);
    cudaDeviceSynchronize();
}

void Measurement::setFirwin(const stdvec_c window)
{
    validateSizeEquals(window.size(), segment_size, "firwin");
    auto tile_window = tile(window, batch_size);
    requireProcessor().setFirwin(tile_window);
}

void Measurement::measure()
{
    Digitizer &active_digitizer = requireDigitizer();
    requireProcessor();
    active_digitizer.prepareFifo(static_cast<unsigned long>(notify_size));
    active_digitizer.launchFifo(static_cast<unsigned long>(notify_size), iters_num, func, true);
    active_digitizer.stopFifo();
    iters_done += iters_num;
}

void Measurement::measureTest()
{
    requireProcessor();
    for (uint32_t i = 0; i < iters_num; i++)
        func(test_input.data());
    iters_done += iters_num;
    // std::cout << "iters done " << iters_done << std::endl; 
}

void Measurement::setTestInput(const std::vector<int8_t> &input)
{
    requireProcessor();
    validateSizeEquals(input.size(), 2 * num_channels * segment_size, "test_input");
    test_input = tile(input, batch_size);
}

corr_t Measurement::getG1Correlator()
{
    dsp &active_processor = requireProcessor();
    int side = active_processor.getResampledTraceLength();
    auto corrs = active_processor.getG1Result();
    return makeCorrelationMatrix(corrs, side);
}

std::tuple<corr_t, corr_t, corr_t> Measurement::getG1OtherCorrelators()
{
    dsp &active_processor = requireProcessor();
    int side = active_processor.getResampledTraceLength();
    auto [reordered_corrs, creation_corrs, annihilation_corrs] = active_processor.getG1OtherResults();

    return {
        makeCorrelationMatrix(reordered_corrs, side),
        makeCorrelationMatrix(creation_corrs, side),
        makeCorrelationMatrix(annihilation_corrs, side)
    };
}

stdvec_c Measurement::getG1CorrelatorFlat()
{
    auto corrs = requireProcessor().getG1Result();
    return postprocess<tcf, output_complex_t>(corrs);
}

std::tuple<stdvec_c, stdvec_c, stdvec_c> Measurement::getG1OtherCorrelatorsFlat()
{
    auto [reordered_corrs, creation_corrs, annihilation_corrs] = requireProcessor().getG1OtherResults();
    return {
        postprocess<tcf, output_complex_t>(reordered_corrs),
        postprocess<tcf, output_complex_t>(creation_corrs),
        postprocess<tcf, output_complex_t>(annihilation_corrs)
    };
}

std::pair<stdvec_c, stdvec_c> Measurement::getAverageField()
{
    dsp &active_processor = requireProcessor();
    int length = active_processor.getResampledTraceLength();
    auto [afs1, afs2] = active_processor.getAverageField();
    output_complex_t X(getIterationsDivisor(), 0.f);

    for (int i = 0; i < length; i++)
    {
        afs1[i] /= X;
        afs2[i] /= X;
    }
    return {afs1, afs2};
}

std::pair<std::complex<float>, std::complex<float>> Measurement::getS21()
{
    auto [s21_1, s21_2] = requireProcessor().getS21();
    output_complex_t X(getIterationsDivisor(), 0.f);
    s21_1 /= X;
    s21_2 /= X;
    return {s21_1, s21_2};
}

stdvec_c Measurement::getCrossPower()
{
    auto cross_power = requireProcessor().getCrossPower();
    return postprocess<tcf, std::complex<float>>(cross_power);
}

stdvec_c Measurement::getCrossSpectrum()
{
    auto cross_spectrum = requireProcessor().getCrossSpectrum();
    return postprocess<tcf, std::complex<float>>(cross_spectrum);
}

void Measurement::setSubtractionTrace(std::vector<stdvec_c> trace)
{
    validateSizeEquals(trace.size(), num_channels, "subtraction_trace");
    dsp &active_processor = requireProcessor();
    const size_t expected_trace_size = static_cast<size_t>(active_processor.getResampledTotalLength());
    hostvec_c average[num_channels];
    for (int i = 0; i < num_channels; i++)
    {
        validateSizeEquals(trace[i].size(), expected_trace_size, "subtraction_trace channel");
        average[i] = trace[i];
    }
    active_processor.setSubtractionTrace(average);
}

// returns newly received data and saved like average_data 
std::vector<stdvec_c> Measurement::getSubtractionData()
{
    std::vector<stdvec_c> subtr_data;
    auto vec = requireProcessor().getCumulativeSubtrData();
    for (int i = 0; i < num_channels; i++)
    {
        subtr_data.push_back(postprocess<tcf, std::complex<float>>(vec[i]));
    }
    
    return subtr_data;
}

// returns traces which were subtracted from data last time
std::vector<stdvec_c> Measurement::getSubtractionTrace()
{
    std::vector<stdvec_c> subtraction_trace;
    requireProcessor().getSubtractionTrace(subtraction_trace);
    return subtraction_trace;
}

template <typename T, typename V>
std::vector<V> Measurement::postprocess(const thrust::host_vector<T> &data)
{
    std::vector<V> result(data.size());
    float divider = getIterationsDivisor();
    thrust::transform(data.cbegin(), data.cend(), result.begin(),
                      [divider](const T &x)
                      { return static_cast<V>(x / divider); });
    return result;
}

float Measurement::getIterationsDivisor() const
{
    return (iters_done > 0) ? static_cast<float>(iters_done) : 1.f;
}

corr_t Measurement::makeCorrelationMatrix(const hostvec_c &data, int side) const
{
    corr_t result(side, trace_t(side));
    tcf divisor(getIterationsDivisor(), 0.f);
    for (int t1 = 0; t1 < side; t1++)
    {
        for (int t2 = 0; t2 < side; t2++)
        {
            tcf value = data[t1 * side + t2] / divisor;
            result[t1][t2] = output_complex_t(value.real(), value.imag());
        }
    }
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
