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

PYBIND11_MODULE(multitaper_measurement, m) {
    py::class_<Measurement>(m, "PsdMeasurer", py::module_local())
        .def(py::init<std::uintptr_t, unsigned long long, int, double, int, const char*>(), OUTPUTON)
        .def("set_calibration", &Measurement::setCalibration, OUTPUTON)
        .def("set_firwin", &Measurement::setFirwin, OUTPUTON)
        .def("measure", &Measurement::measure, OUTPUTON)
        .def("measure_with_coil", &Measurement::measureWithCoil, OUTPUTON)
        .def("get_mean_field", &Measurement::getMeanField, OUTPUTON)
        .def("get_mean_power", &Measurement::getMeanPower, OUTPUTON)
        //.def("get_correlator", &Measurement::getCorrelator, OUTPUTON)
        .def("get_psd", &Measurement::getPSD, OUTPUTON)
        .def("get_data_spectrum", &Measurement::getDataSpectrum, OUTPUTON)
        .def("get_noise_spectrum", &Measurement::getNoiseSpectrum, OUTPUTON)
        //.def("get_periodogram", &Measurement::getPeriodogram, OUTPUTON)
        //.def("get_raw_correlator", &Measurement::getRawCorrelator, OUTPUTON)
        .def("reset", &Measurement::reset, OUTPUTON)
        .def("reset_output", &Measurement::resetOutput, OUTPUTON)
        .def("free", &Measurement::free, OUTPUTON)
        .def("measure_test", &Measurement::measureTest, OUTPUTON)
        .def("set_test_input", &Measurement::setTestInput, OUTPUTON)
        .def("set_subtraction_trace", &Measurement::setSubtractionTrace, OUTPUTON)
        .def("get_subtraction_trace", &Measurement::getSubtractionTrace)
        .def("get_subtraction_data", &Measurement::getSubtractionData, OUTPUTON)
        .def("get_subtraction_noise", &Measurement::getSubtractionNoise, OUTPUTON)
        .def("set_amplitude", &Measurement::setAmplitude, OUTPUTON)
        .def("set_currents", &Measurement::setCurrents, OUTPUTON)
        .def("set_intermediate_frequency", &Measurement::setIntermediateFrequency, OUTPUTON)
        .def("set_averages_number", &Measurement::setAveragesNumber, OUTPUTON)
        .def("get_total_length", &Measurement::getTotalLength, OUTPUTON)
        .def("get_trace_length", &Measurement::getTraceLength, OUTPUTON)
        .def("get_out_size", &Measurement::getOutSize, OUTPUTON)
        .def("get_notify_size", &Measurement::getNotifySize, OUTPUTON)
        .def("set_dpss_tapers", &Measurement::setTapers, OUTPUTON)
        .def("get_dpss_tapers", &Measurement::getDPSSTapers, OUTPUTON);
}
