//
// Created by andrei on 4/13/21.
//
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include "measurement.cuh"

namespace py = pybind11;

using output_and_gil_guard = py::call_guard<py::scoped_ostream_redirect,
                                            py::scoped_estream_redirect,
                                            py::gil_scoped_release>;

using namespace pybind11::literals;


PYBIND11_MODULE(multitaper_measurement, m)
{
    py::class_<Measurement>(m, "PsdMeasurer", py::module_local())
        .def(py::init<std::uintptr_t, unsigned long long, int, double, int, int, const char *>(), output_and_gil_guard())
        .def("set_amplitude", &Measurement::setAmplitude, output_and_gil_guard())
        .def("set_averages_number", &Measurement::setAveragesNumber, output_and_gil_guard())
        .def("set_intermediate_frequency", &Measurement::setIntermediateFrequency, output_and_gil_guard())
        .def("set_currents", &Measurement::setCurrents, output_and_gil_guard())
        .def("set_calibration", &Measurement::setCalibration, output_and_gil_guard())
        .def("set_firwin", &Measurement::setFirwin, output_and_gil_guard())
        .def("set_test_input", &Measurement::setTestInput, output_and_gil_guard())
        .def("measure", &Measurement::measure, output_and_gil_guard())
        .def("measure_with_coil", &Measurement::measureWithCoil, output_and_gil_guard())
        .def("measure_test", &Measurement::measureTest, output_and_gil_guard())
        .def("get_mean_field", &Measurement::getMeanField, output_and_gil_guard())
        .def("get_mean_power", &Measurement::getMeanPower, output_and_gil_guard())
        .def("get_psd", &Measurement::getPSD, output_and_gil_guard())
        .def("get_data_spectrum", &Measurement::getDataSpectrum, output_and_gil_guard())
        .def("get_noise_spectrum", &Measurement::getNoiseSpectrum, output_and_gil_guard())
        .def("get_periodogram", &Measurement::getPeriodogram, output_and_gil_guard())
        .def("reset", &Measurement::reset, output_and_gil_guard())
        .def("reset_output", &Measurement::resetOutput, output_and_gil_guard())
        .def("free", &Measurement::free, output_and_gil_guard())
        .def("set_subtraction_traces", &Measurement::setSubtractionTraces, output_and_gil_guard())
        .def("get_subtraction_traces", &Measurement::getSubtractionTraces, output_and_gil_guard())
        .def("get_accumulated_subtraction_data", &Measurement::getAccumulatedSubstractionData, output_and_gil_guard())
        .def("get_total_length", &Measurement::getTotalLength, output_and_gil_guard())
        .def("get_trace_length", &Measurement::getTraceLength, output_and_gil_guard())
        .def("get_notify_size", &Measurement::getNotifySize, output_and_gil_guard())
        .def("set_dpss_tapers", &Measurement::setTapers, output_and_gil_guard())
        .def("get_dpss_tapers", &Measurement::getDPSSTapers, output_and_gil_guard());
}
