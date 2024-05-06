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

PYBIND11_MODULE(filtered_psd2, m)
{
    py::class_<Measurement>(m, "PSDFiltMeasurer", py::module_local())
        .def(py::init<std::uintptr_t, unsigned long long, unsigned long long, double, int, const char *>(), output_and_gil_guard())
        // .def(py::init<unsigned long long, unsigned long long, long, float, int, int>(), output_and_gil_guard()) // for test inputs with out digitizer
        .def("set_calibration", &Measurement::setCalibration, output_and_gil_guard())
        .def("set_firwin", py::overload_cast<float, float>(&Measurement::setFirwin), output_and_gil_guard(), "Set rectangular window")
        .def("set_firwin", py::overload_cast<const stdvec_c>(&Measurement::setFirwin), output_and_gil_guard(), "Set custom window")
        .def("set_additional_firwins", &Measurement::setAdditionalFirwins, output_and_gil_guard(), "Set filters for the used micromodule")
        // .def("set_central_firwin", py::overload_cast<float, float>(&Measurement::setCentralPeakWin), output_and_gil_guard(), "Set rectangular window")
        // .def("set_central_firwin", py::overload_cast<const stdvec_c>(&Measurement::setCentralPeakWin), output_and_gil_guard(), "Set custom window")
        // .def("set_correlation_firwin", py::overload_cast<std::pair<float, float>, std::pair<float, float>>(&Measurement::setCorrelationFirwin), output_and_gil_guard())
        // .def("set_correlation_firwin", py::overload_cast<const stdvec_c, const stdvec_c>(&Measurement::setCorrelationFirwin), output_and_gil_guard())
        .def("measure", &Measurement::measure, output_and_gil_guard())
        .def("measure_with_coil", &Measurement::measureWithCoil, output_and_gil_guard())
        .def("get_g1_filt", &Measurement::getG1Filt, output_and_gil_guard())
        .def("get_g1_filt_conj", &Measurement::getG1FiltConj, output_and_gil_guard())
        .def("get_g1_without_cp", &Measurement::getG1WithoutCP, output_and_gil_guard())
        .def("get_g2_filt", &Measurement::getG2Filt, output_and_gil_guard())
        .def("get_interference", &Measurement::getInterference, output_and_gil_guard())
        // .def("get_PSD", &Measurement::getPSD, output_and_gil_guard())
        .def("set_corr_downconvert_freqs", &Measurement::setCorrDowncovertCoeffs, output_and_gil_guard())
        .def("reset", &Measurement::reset, output_and_gil_guard())
        .def("reset_output", &Measurement::resetOutput, output_and_gil_guard())
        .def("free", &Measurement::free, output_and_gil_guard())
        .def("measure_test", &Measurement::measureTest, output_and_gil_guard())
        .def("set_test_input", &Measurement::setTestInput, output_and_gil_guard())
        .def("set_subtraction_trace", &Measurement::setSubtractionTrace, output_and_gil_guard())
        .def("get_subtraction_trace", &Measurement::getSubtractionTrace, output_and_gil_guard())
        .def("get_average_data", &Measurement::getAverageData, output_and_gil_guard())
        .def("set_amplitude", &Measurement::setAmplitude, output_and_gil_guard())
        .def("set_currents", &Measurement::setCurrents, output_and_gil_guard())
        .def("set_intermediate_frequency", &Measurement::setIntermediateFrequency, output_and_gil_guard())
        .def("set_averages_number", &Measurement::setAveragesNumber, output_and_gil_guard())
        .def("get_total_length", &Measurement::getTotalLength, output_and_gil_guard())
        .def("get_trace_length", &Measurement::getTraceLength, output_and_gil_guard())
        .def("get_out_size", &Measurement::getOutSize, output_and_gil_guard())
        .def("get_notify_size", &Measurement::getNotifySize, output_and_gil_guard())
        // .def("get_firwins", &Measurement::getFirwins, output_and_gil_guard())
        .def("get_real_results", &Measurement::getRealResults, output_and_gil_guard())
        .def("get_complex_results", &Measurement::getComplexResults, output_and_gil_guard());
}
