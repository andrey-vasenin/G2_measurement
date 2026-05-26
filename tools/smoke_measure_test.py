from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np


def _add_dll_directories() -> None:
    cuda_path = os.environ.get("CUDA_PATH")
    candidates: list[Path] = []
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin" / "x64")
        candidates.append(Path(cuda_path) / "bin")
    candidates.append(Path(r"C:\Windows\System32"))

    for candidate in candidates:
        if candidate.exists():
            try:
                os.add_dll_directory(str(candidate))
            except (AttributeError, OSError):
                pass


def _find_module(build_dir: Path) -> Path:
    candidates = sorted(build_dir.glob("AverageField*.pyd"))
    candidates.extend(sorted((build_dir / "Release").glob("AverageField*.pyd")))
    if not candidates:
        raise FileNotFoundError(f"No AverageField*.pyd found in {build_dir}")
    return candidates[0]


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("AverageField", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_values(segment: int) -> tuple[list[complex], list[complex]]:
    ch1: list[complex] = []
    ch2: list[complex] = []
    for i in range(segment):
        ch1.append(complex((i % 7) - 3, ((2 * i) % 9) - 4))
        ch2.append(complex(((3 * i + 1) % 11) - 5, ((5 * i + 2) % 13) - 6))
    return ch1, ch2


def _interleaved_int8_input(ch1: list[complex], ch2: list[complex]) -> list[int]:
    raw: list[int] = []
    for a, b in zip(ch1, ch2):
        raw.extend([int(a.real), int(a.imag), int(b.real), int(b.imag)])
    return raw


def _downsample(values: list[complex], factor: int) -> list[complex]:
    if factor == 1:
        return list(values)
    if factor not in (2, 4):
        raise ValueError(f"Unsupported second oversampling in smoke test: {factor}")
    if len(values) % factor != 0:
        raise ValueError(f"Trace length {len(values)} is not divisible by {factor}")
    return [
        sum(values[i : i + factor], 0j) / factor
        for i in range(0, len(values), factor)
    ]


def _assert_close(actual: complex, expected: complex, label: str, tol: float) -> None:
    scale = max(1.0, abs(expected))
    if abs(actual - expected) > tol * scale:
        raise AssertionError(f"{label}: got {actual!r}, expected {expected!r}")


def _assert_sequence_close(actual: list[complex], expected: list[complex], label: str, tol: float) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: length {len(actual)} != expected {len(expected)}")
    for idx, (a, e) in enumerate(zip(actual, expected)):
        _assert_close(complex(a), complex(e), f"{label}[{idx}]", tol)


def _assert_finite(values, label: str) -> None:
    for idx, value in enumerate(values):
        z = complex(value)
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            raise AssertionError(f"{label}[{idx}] is not finite: {value!r}")


def _assert_matrix_shape(matrix, side: int, label: str) -> None:
    if len(matrix) != side:
        raise AssertionError(f"{label}: row count {len(matrix)} != expected {side}")
    for row_idx, row in enumerate(matrix):
        if len(row) != side:
            raise AssertionError(f"{label}[{row_idx}]: column count {len(row)} != expected {side}")


def _assert_complex64_array(actual, shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.asarray(actual)
    if arr.shape != shape:
        raise AssertionError(f"{label}: shape {arr.shape} != expected {shape}")
    if arr.dtype != np.dtype(np.complex64):
        raise AssertionError(f"{label}: dtype {arr.dtype} != expected complex64")
    _assert_finite(arr.ravel(), label)
    return arr


def _assert_unavailable(func, label: str) -> None:
    try:
        func()
    except RuntimeError as exc:
        if "result_mode" not in str(exc):
            raise AssertionError(f"{label}: unexpected RuntimeError: {exc}") from exc
        return
    raise AssertionError(f"{label}: getter unexpectedly succeeded")


def _assert_constructor_invalid(module, args: tuple, label: str, expected: str) -> None:
    try:
        module.AverageFieldMeasurer(*args)
    except RuntimeError as exc:
        if expected not in str(exc):
            raise AssertionError(f"{label}: unexpected RuntimeError: {exc}") from exc
        return
    raise AssertionError(f"{label}: constructor unexpectedly succeeded")


def _assert_runtime_error(func, label: str, expected: str) -> None:
    try:
        func()
    except RuntimeError as exc:
        if expected not in str(exc):
            raise AssertionError(f"{label}: unexpected RuntimeError: {exc}") from exc
        return
    raise AssertionError(f"{label}: call unexpectedly succeeded")


def _run_invalid_constructor_cases(module) -> None:
    cases = [
        ((2, 0, 16, 1, 1, "average"), "zero batch", "batch"),
        ((2, 2, 0, 1, 1, "average"), "zero segment", "segment"),
        ((2, 2, 16, 0, 1, "average"), "zero digitizer oversampling", "digitizer_oversampling"),
        ((2, 2, 16, 1, 3, "average"), "unsupported second oversampling", "second_oversampling"),
        ((2, 2, 18, 1, 4, "average"), "non-divisible segment", "divisible"),
        ((3, 2, 16, 1, 1, "average"), "non-divisible averages", "averages"),
        ((2, 2, 16, 1, 1, "bad_mode"), "unsupported result mode", "Unsupported result_mode"),
    ]
    for args, label, expected in cases:
        _assert_constructor_invalid(module, args, label, expected)
    print("invalid constructor checks passed")


def _run_invalid_setter_cases(module) -> None:
    measurer = module.AverageFieldMeasurer(2, 2, 16, 1, 1, "average")
    freed = False
    try:
        _assert_runtime_error(lambda: measurer.set_calibration(2, 1.0, 0.0, 0.0, 0.0), "bad calibration channel", "line_num")
        _assert_runtime_error(lambda: measurer.set_firwin([1.0 + 0.0j] * 15), "bad FIR length", "firwin")
        _assert_runtime_error(lambda: measurer.set_test_input([0] * 63), "bad test input length", "test_input")
        _assert_runtime_error(lambda: measurer.set_subtraction_trace([[0.0 + 0.0j] * 16]), "bad subtraction trace count", "subtraction_trace")
        _assert_runtime_error(measurer.measure, "measure without digitizer", "digitizer")
        measurer.free()
        freed = True
        _assert_runtime_error(measurer.get_result_mode, "getter after free", "freed")
        _assert_runtime_error(lambda: measurer.set_averages_number(2), "setter after free", "freed")
        _assert_runtime_error(measurer.measure_test, "measure_test after free", "freed")
        measurer.free()
    finally:
        if not freed:
            measurer.free()
    print("invalid setter checks passed")


def _run_case(
    module,
    segment: int,
    batch: int,
    averages: int,
    second_oversampling: int,
    result_mode: str,
    tol: float,
) -> None:
    if averages % batch != 0:
        raise ValueError("averages must be divisible by batch")

    ch1, ch2 = _trace_values(segment)
    expected_ch1 = _downsample(ch1, second_oversampling)
    expected_ch2 = _downsample(ch2, second_oversampling)
    out_len = len(expected_ch1)

    measurer = module.AverageFieldMeasurer(
        averages,
        batch,
        segment,
        1,
        second_oversampling,
        result_mode,
    )
    try:
        if measurer.get_result_mode() != result_mode:
            raise AssertionError("get_result_mode returned an unexpected value")
        measurer.set_amplitude(128)
        measurer.set_calibration(0, 1.0, 0.0, 0.0, 0.0)
        measurer.set_calibration(1, 1.0, 0.0, 0.0, 0.0)
        measurer.set_firwin([1.0 + 0.0j] * segment)
        measurer.set_intermediate_frequency(0.0)
        measurer.set_test_input(_interleaved_int8_input(ch1, ch2))
        measurer.measure_test()

        if measurer.get_total_length() != segment * batch:
            raise AssertionError("get_total_length returned an unexpected value")
        if measurer.get_trace_length() != segment:
            raise AssertionError("get_trace_length returned an unexpected value")
        if measurer.get_resampled_trace_length() != out_len:
            raise AssertionError("get_resampled_trace_length returned an unexpected value")
        if measurer.get_out_size() != out_len * out_len:
            raise AssertionError("get_out_size returned an unexpected value")
        if measurer.get_notify_size() != 4 * segment * batch:
            raise AssertionError("get_notify_size returned an unexpected value")

        average_ch1, average_ch2 = measurer.get_average_field()
        _assert_sequence_close(list(average_ch1), expected_ch1, "average_field[0]", tol)
        _assert_sequence_close(list(average_ch2), expected_ch2, "average_field[1]", tol)

        average_array = _assert_complex64_array(measurer.get_average_field_array(), (2, out_len), "average_field_array")
        _assert_sequence_close(average_array[0].tolist(), expected_ch1, "average_field_array[0]", tol)
        _assert_sequence_close(average_array[1].tolist(), expected_ch2, "average_field_array[1]", tol)

        s21_1, s21_2 = measurer.get_s21()
        _assert_close(complex(s21_1), sum(expected_ch1, 0j) / out_len, "s21[0]", tol)
        _assert_close(complex(s21_2), sum(expected_ch2, 0j) / out_len, "s21[1]", tol)

        s21_array = _assert_complex64_array(measurer.get_s21_array(), (2,), "s21_array")
        _assert_close(complex(s21_array[0]), sum(expected_ch1, 0j) / out_len, "s21_array[0]", tol)
        _assert_close(complex(s21_array[1]), sum(expected_ch2, 0j) / out_len, "s21_array[1]", tol)

        if result_mode == "average":
            _assert_unavailable(measurer.get_g1_correlator, "get_g1_correlator")
            _assert_unavailable(measurer.get_g1_correlator_array, "get_g1_correlator_array")
            _assert_unavailable(measurer.get_g1_other_correlators, "get_g1_other_correlators")
            _assert_unavailable(measurer.get_g1_other_correlators_array, "get_g1_other_correlators_array")
            _assert_unavailable(measurer.get_cross_power, "get_cross_power")
            _assert_unavailable(measurer.get_cross_power_array, "get_cross_power_array")
            _assert_unavailable(measurer.get_cross_spectrum, "get_cross_spectrum")
            _assert_unavailable(measurer.get_cross_spectrum_array, "get_cross_spectrum_array")
        else:
            g1 = measurer.get_g1_correlator()
            _assert_matrix_shape(g1, out_len, "g1")
            g1_array = _assert_complex64_array(measurer.get_g1_correlator_array(), (out_len, out_len), "g1_array")
            for idx in range(out_len):
                expected_diag = expected_ch1[idx] * expected_ch2[idx].conjugate()
                _assert_close(complex(g1[idx][idx]), expected_diag, f"g1[{idx}][{idx}]", tol)
                _assert_close(complex(g1_array[idx, idx]), expected_diag, f"g1_array[{idx},{idx}]", tol)

            if result_mode == "average_g1":
                _assert_unavailable(measurer.get_g1_other_correlators, "get_g1_other_correlators")
                _assert_unavailable(measurer.get_g1_other_correlators_array, "get_g1_other_correlators_array")
                _assert_unavailable(measurer.get_cross_power, "get_cross_power")
                _assert_unavailable(measurer.get_cross_power_array, "get_cross_power_array")
                _assert_unavailable(measurer.get_cross_spectrum, "get_cross_spectrum")
                _assert_unavailable(measurer.get_cross_spectrum_array, "get_cross_spectrum_array")
            else:
                cross_power = list(measurer.get_cross_power())
                expected_cross_power = [a.conjugate() * b for a, b in zip(expected_ch1, expected_ch2)]
                _assert_sequence_close(cross_power, expected_cross_power, "cross_power", tol)

                cross_power_array = _assert_complex64_array(measurer.get_cross_power_array(), (out_len,), "cross_power_array")
                _assert_sequence_close(cross_power_array.tolist(), expected_cross_power, "cross_power_array", tol)

                other_g1 = measurer.get_g1_other_correlators()
                if len(other_g1) != 3:
                    raise AssertionError(f"get_g1_other_correlators returned {len(other_g1)} matrices")
                for idx, matrix in enumerate(other_g1):
                    _assert_matrix_shape(matrix, out_len, f"g1_other[{idx}]")

                _assert_complex64_array(measurer.get_g1_other_correlators_array(), (3, out_len, out_len), "g1_other_array")

                cross_spectrum = list(measurer.get_cross_spectrum())
                if len(cross_spectrum) != out_len:
                    raise AssertionError("get_cross_spectrum returned an unexpected length")
                _assert_finite(cross_spectrum, "cross_spectrum")
                _assert_complex64_array(measurer.get_cross_spectrum_array(), (out_len,), "cross_spectrum_array")

        subtraction_data = measurer.get_subtraction_data()
        if len(subtraction_data) != 2:
            raise AssertionError("get_subtraction_data should return two traces")
        for idx, trace in enumerate(subtraction_data):
            if len(trace) != out_len * batch:
                raise AssertionError(f"subtraction_data[{idx}] returned an unexpected length")
            _assert_finite(trace, f"subtraction_data[{idx}]")
        _assert_complex64_array(measurer.get_subtraction_data_array(), (2, out_len * batch), "subtraction_data_array")

        subtraction_trace = measurer.get_subtraction_trace()
        if len(subtraction_trace) != 2:
            raise AssertionError("get_subtraction_trace should return two traces")
        for idx, trace in enumerate(subtraction_trace):
            if len(trace) != out_len * batch:
                raise AssertionError(f"subtraction_trace[{idx}] returned an unexpected length")
            _assert_finite(trace, f"subtraction_trace[{idx}]")
        _assert_complex64_array(measurer.get_subtraction_trace_array(), (2, out_len * batch), "subtraction_trace_array")
    finally:
        measurer.free()

    print(
        f"measure_test passed: segment={segment}, batch={batch}, "
        f"averages={averages}, second_oversampling={second_oversampling}, result_mode={result_mode}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-hardware synthetic AverageField GPU regression test.")
    parser.add_argument("--build-dir", default="build/windows-qom-ninja", help="CMake build directory")
    parser.add_argument("--module", default=None, help="Explicit path to AverageField*.pyd")
    parser.add_argument("--segment", type=int, default=16, help="Synthetic segment length")
    parser.add_argument("--batch", type=int, default=2, help="Synthetic batch size")
    parser.add_argument("--averages", type=int, default=2, help="Synthetic averages count")
    parser.add_argument(
        "--second-oversampling",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Second-oversampling factors to exercise.",
    )
    parser.add_argument(
        "--result-mode",
        nargs="+",
        default=["average", "average_g1", "all_correlators"],
        choices=["average", "average_g1", "all_correlators"],
        help="Result modes to exercise.",
    )
    parser.add_argument("--tolerance", type=float, default=2e-3, help="Relative numerical tolerance")
    args = parser.parse_args()

    _add_dll_directories()
    module_path = Path(args.module) if args.module else _find_module(Path(args.build_dir))
    module_path = module_path.resolve()
    module = _load_module(module_path)
    print(f"Loaded: {module_path}")

    _run_invalid_constructor_cases(module)
    _run_invalid_setter_cases(module)

    for second_oversampling in args.second_oversampling:
        for result_mode in args.result_mode:
            _run_case(module, args.segment, args.batch, args.averages, second_oversampling, result_mode, args.tolerance)


if __name__ == "__main__":
    main()
