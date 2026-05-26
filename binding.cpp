//
// Created by andrei on 4/13/21.
//
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include "measurement.cuh"

namespace py = pybind11;
using output_and_gil_guard = py::call_guard<py::scoped_ostream_redirect,
                                            py::scoped_estream_redirect,
                                            py::gil_scoped_release>;

using namespace pybind11::literals;

namespace
{
size_t shapeElementCount(const std::vector<py::ssize_t> &shape)
{
    size_t count = 1;
    for (py::ssize_t dim : shape)
    {
        if (dim < 0)
            throw std::runtime_error("array shape contains a negative dimension");
        const size_t dim_size = static_cast<size_t>(dim);
        if (dim_size != 0 && count > std::numeric_limits<size_t>::max() / dim_size)
            throw std::runtime_error("array shape is too large");
        count *= dim_size;
    }
    return count;
}

py::array_t<output_complex_t> makeComplexArray(const stdvec_c &values, std::vector<py::ssize_t> shape)
{
    if (shapeElementCount(shape) != values.size())
        throw std::runtime_error("internal error: array shape does not match result size");

    py::array_t<output_complex_t> array(shape);
    if (!values.empty())
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(output_complex_t));
    return array;
}

stdvec_c stackEqualLengthRows(const std::vector<stdvec_c> &rows, const char *name)
{
    if (rows.empty())
        return {};

    const size_t row_size = rows.front().size();
    stdvec_c stacked;
    stacked.reserve(row_size * rows.size());
    for (size_t row_idx = 0; row_idx < rows.size(); row_idx++)
    {
        if (rows[row_idx].size() != row_size)
            throw std::runtime_error(std::string(name) + " rows have inconsistent lengths");
        stacked.insert(stacked.end(), rows[row_idx].begin(), rows[row_idx].end());
    }
    return stacked;
}
}

PYBIND11_MODULE(AverageField, m)
{
    py::class_<Measurement>(m, "AverageFieldMeasurer", py::module_local())
        .def(py::init<std::uintptr_t, uint64_t, uint64_t, int, std::string, std::string>(),
             "digitizer_handle"_a, "averages"_a, "batch"_a, "second_oversampling"_a, "result_mode"_a = "average_g1",
             "channel_layout"_a = "two_complex",
             output_and_gil_guard())
        .def(py::init<uint64_t, uint64_t, long, int, int, std::string, std::string>(),
             "averages"_a, "batch"_a, "segment"_a, "digitizer_oversampling"_a, "second_oversampling"_a, "result_mode"_a = "average_g1",
             "channel_layout"_a = "two_complex",
             output_and_gil_guard())
        .def("set_calibration", &Measurement::setCalibration, output_and_gil_guard())
        .def("set_firwin", py::overload_cast<float, float>(&Measurement::setFirwin), output_and_gil_guard(), "Set rectangular window")
        .def("set_firwin", py::overload_cast<const stdvec_c>(&Measurement::setFirwin), output_and_gil_guard(), "Set custom window")
        .def("measure", &Measurement::measure, output_and_gil_guard())
        .def("start_fifo", &Measurement::startFifo, output_and_gil_guard())
        .def("stop_fifo", &Measurement::stopFifo, output_and_gil_guard())
        .def("is_fifo_active", &Measurement::isFifoActive, output_and_gil_guard())
        .def("measure_batches", &Measurement::measureBatches, output_and_gil_guard())
        .def("get_g1_correlator", &Measurement::getG1Correlator, output_and_gil_guard())
        .def("get_g1_other_correlators", &Measurement::getG1OtherCorrelators, output_and_gil_guard())
        .def("get_g1_correlator_array", [](Measurement &measurement)
        {
            int side = 0;
            stdvec_c values;
            {
                py::gil_scoped_release release;
                side = measurement.getResampledTraceLength();
                values = measurement.getG1CorrelatorFlat();
            }
            return makeComplexArray(values, {side, side});
        })
        .def("get_g1_other_correlators_array", [](Measurement &measurement)
        {
            int side = 0;
            std::tuple<stdvec_c, stdvec_c, stdvec_c> correlators;
            {
                py::gil_scoped_release release;
                side = measurement.getResampledTraceLength();
                correlators = measurement.getG1OtherCorrelatorsFlat();
            }
            std::vector<stdvec_c> rows;
            rows.reserve(3);
            rows.push_back(std::move(std::get<0>(correlators)));
            rows.push_back(std::move(std::get<1>(correlators)));
            rows.push_back(std::move(std::get<2>(correlators)));
            return makeComplexArray(stackEqualLengthRows(rows, "g1_other_correlators"), {3, side, side});
        })
        .def("get_average_field", &Measurement::getAverageField, output_and_gil_guard())
        .def("get_average_field_array", [](Measurement &measurement)
        {
            std::pair<stdvec_c, stdvec_c> average;
            {
                py::gil_scoped_release release;
                average = measurement.getAverageField();
            }
            std::vector<stdvec_c> rows;
            rows.reserve(2);
            rows.push_back(std::move(average.first));
            if (!average.second.empty())
                rows.push_back(std::move(average.second));
            const py::ssize_t length = static_cast<py::ssize_t>(rows.empty() ? 0 : rows.front().size());
            return makeComplexArray(stackEqualLengthRows(rows, "average_field"), {static_cast<py::ssize_t>(rows.size()), length});
        })
        .def("get_s21", &Measurement::getS21, output_and_gil_guard())
        .def("get_s21_array", [](Measurement &measurement)
        {
            std::pair<output_complex_t, output_complex_t> s21;
            {
                py::gil_scoped_release release;
                s21 = measurement.getS21();
            }
            stdvec_c values{s21.first};
            if (measurement.getComplexFieldCount() == 2)
                values.push_back(s21.second);
            return makeComplexArray(values, {static_cast<py::ssize_t>(values.size())});
        })
        .def("get_cross_power", &Measurement::getCrossPower, output_and_gil_guard())
        .def("get_cross_power_array", [](Measurement &measurement)
        {
            stdvec_c cross_power;
            {
                py::gil_scoped_release release;
                cross_power = measurement.getCrossPower();
            }
            return makeComplexArray(cross_power, {static_cast<py::ssize_t>(cross_power.size())});
        })
        .def("get_cross_spectrum", &Measurement::getCrossSpectrum, output_and_gil_guard())
        .def("get_cross_spectrum_array", [](Measurement &measurement)
        {
            stdvec_c cross_spectrum;
            {
                py::gil_scoped_release release;
                cross_spectrum = measurement.getCrossSpectrum();
            }
            return makeComplexArray(cross_spectrum, {static_cast<py::ssize_t>(cross_spectrum.size())});
        })
        .def("set_corr_downconvert_freqs", &Measurement::setCorrDowncovertCoeffs, output_and_gil_guard())
        .def("reset", &Measurement::reset, output_and_gil_guard())
        .def("reset_output", &Measurement::resetOutput, output_and_gil_guard())
        .def("free", &Measurement::free, output_and_gil_guard())
        .def("measure_test", &Measurement::measureTest, output_and_gil_guard())
        .def("measure_test_batches", &Measurement::measureTestBatches, output_and_gil_guard())
        .def("set_test_input", &Measurement::setTestInput, output_and_gil_guard())
        .def("set_subtraction_trace", &Measurement::setSubtractionTrace, output_and_gil_guard())
        .def("get_subtraction_trace", &Measurement::getSubtractionTrace, output_and_gil_guard())
        .def("get_subtraction_trace_array", [](Measurement &measurement)
        {
            std::vector<stdvec_c> traces;
            {
                py::gil_scoped_release release;
                traces = measurement.getSubtractionTrace();
            }
            const py::ssize_t length = static_cast<py::ssize_t>(traces.empty() ? 0 : traces.front().size());
            return makeComplexArray(stackEqualLengthRows(traces, "subtraction_trace"), {static_cast<py::ssize_t>(traces.size()), length});
        })
        .def("get_subtraction_data", &Measurement::getSubtractionData, output_and_gil_guard())
        .def("get_subtraction_data_array", [](Measurement &measurement)
        {
            std::vector<stdvec_c> traces;
            {
                py::gil_scoped_release release;
                traces = measurement.getSubtractionData();
            }
            const py::ssize_t length = static_cast<py::ssize_t>(traces.empty() ? 0 : traces.front().size());
            return makeComplexArray(stackEqualLengthRows(traces, "subtraction_data"), {static_cast<py::ssize_t>(traces.size()), length});
        })
        .def("set_amplitude", &Measurement::setAmplitude, output_and_gil_guard())
        .def("set_intermediate_frequency", &Measurement::setIntermediateFrequency, output_and_gil_guard())
        .def("set_averages_number", &Measurement::setAveragesNumber, output_and_gil_guard())
        .def("get_total_length", &Measurement::getTotalLength, output_and_gil_guard())
        .def("get_trace_length", &Measurement::getTraceLength, output_and_gil_guard())
        .def("get_resampled_trace_length", &Measurement::getResampledTraceLength, output_and_gil_guard())
        .def("get_result_mode", &Measurement::getResultMode, output_and_gil_guard())
        .def("get_channel_layout", &Measurement::getChannelLayout, output_and_gil_guard())
        .def("get_complex_field_count", &Measurement::getComplexFieldCount, output_and_gil_guard())
        .def("get_physical_channel_count", &Measurement::getPhysicalChannelCount, output_and_gil_guard())
        .def("get_batches_total", &Measurement::getBatchesTotal, output_and_gil_guard())
        .def("get_batches_done", &Measurement::getBatchesDone, output_and_gil_guard())
        .def("get_batches_remaining", &Measurement::getBatchesRemaining, output_and_gil_guard())
        .def("get_averages_total", &Measurement::getAveragesTotal, output_and_gil_guard())
        .def("get_averages_done", &Measurement::getAveragesDone, output_and_gil_guard())
        .def("get_out_size", &Measurement::getOutSize, output_and_gil_guard())
        .def("get_notify_size", &Measurement::getNotifySize, output_and_gil_guard());
}
