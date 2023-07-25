//
// Created by andrei on 4/13/21.
//
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include "measurement.cuh"

#define OUTPUTON py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect, py::gil_scoped_release>()

using namespace pybind11::literals;

namespace py = pybind11;

PYBIND11_MODULE(g2measurement, m) {
    py::class_<Measurement>(m, "G2Measurer", py::module_local())
        .def(py::init<std::uintptr_t, unsigned long long, int, float>(), OUTPUTON)
        .def("set_calibration", &Measurement::setCalibration, OUTPUTON)
        .def("set_firwin", &Measurement::setFirwin, OUTPUTON)
        .def("measure", &Measurement::measure, OUTPUTON)
        .def("get_mean_field", &Measurement::getMeanField, OUTPUTON)
        .def("get_mean_power", &Measurement::getMeanPower, OUTPUTON)
        .def("get_correlator", &Measurement::getCorrelator, OUTPUTON)
        .def("get_raw_correlator", &Measurement::getRawCorrelator, OUTPUTON)
        .def("reset", &Measurement::reset, OUTPUTON)
        .def("reset_output", &Measurement::resetOutput, OUTPUTON)
        .def("get_counter", &Measurement::getCounter, OUTPUTON)
        .def("free", &Measurement::free, OUTPUTON)
        .def("measure_test", &Measurement::measureTest, OUTPUTON)
        .def("set_test_input", &Measurement::setTestInput, OUTPUTON)
        .def("set_subtraction_trace", &Measurement::setSubtractionTrace, OUTPUTON)
        .def("get_subtraction_trace", &Measurement::getSubtractionTrace, OUTPUTON)
        .def("set_amplitude", &Measurement::setAmplitude, OUTPUTON)
        .def("set_intermediate_frequency", &Measurement::setIntermediateFrequency, OUTPUTON)
        .def("set_averages_number", &Measurement::setAveragesNumber, OUTPUTON)
        .def("get_total_length", &Measurement::getTotalLength, OUTPUTON)
        .def("get_trace_length", &Measurement::getTraceLength, OUTPUTON)
        .def("get_out_size", &Measurement::getOutSize, OUTPUTON)
        .def("get_notify_size", &Measurement::getNotifySize, OUTPUTON);
}
