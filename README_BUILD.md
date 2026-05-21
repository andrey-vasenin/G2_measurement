# AverageField Build and Smoke Test

This project is a Windows/CUDA pybind11 extension for the Spectrum M4x digitizer. The supported MeasurementPC workflow is VSCode + CMake presets launched from a prepared `cmd.exe`.

## MeasurementPC Prerequisites

Known-good environment:

- Windows 11 x64
- Visual Studio 2022 Community / MSVC 19.44
- CUDA Toolkit 13.0 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- Conda env `qom` at `C:\Users\Qop\miniconda3\envs\qom`
- Python ABI `.cp313-win_amd64.pyd`
- Spectrum SDK headers/import lib in `c_header`
- Spectrum runtime DLL in `C:\Windows\System32\spcm_win64.dll`
- VISA at `C:\Program Files\IVI Foundation\VISA\Win64`

Install the Python-side build helpers once:

```bat
C:\Users\Qop\miniconda3\Scripts\activate.bat qom
python -m pip install pybind11 cmake ninja
```

## Configure Shell

Open normal Command Prompt, then run:

```bat
C:\Users\Qop\miniconda3\Scripts\activate.bat qom
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
cd C:\Users\Qop\AverageField
```

Check the tools:

```bat
where python
where cmake
where ninja
where cl
where nvcc
python -c "import numpy, pybind11; print(numpy.__version__, numpy.get_include()); print(pybind11.__version__, pybind11.get_cmake_dir())"
```

## VSCode Workflow

Launch VSCode from the prepared prompt so it inherits `qom`, MSVC, and CUDA paths:

```bat
code .
```

In VSCode CMake Tools:

1. Select configure preset `Windows qom Ninja`.
2. Configure.
3. Build preset `Build Windows qom Release`.

Equivalent command-line build:

```bat
cmake --preset windows-qom-ninja
cmake --build --preset windows-qom-release --verbose
```

The built extension should appear under:

```text
build\windows-qom-ninja\AverageField.cp313-win_amd64.pyd
```

## Import Smoke Test

The recommended no-hardware validation is the CMake smoke-import build preset. It builds `AverageField` if needed, then imports the produced `.pyd` directly:

```bat
cmake --build --preset windows-qom-smoke-import --verbose
```

You can also run the smoke test through CTest after building:

```bat
ctest --preset windows-qom-smoke
```

For direct debugging, the underlying script is still available:

```bat
python tools\smoke_import.py --build-dir build\windows-qom-ninja
```

All three routes load the built `.pyd` and print or validate the available `AverageFieldMeasurer` methods. They do not touch digitizer hardware.

## Optional Deploy to QO Notebook Package

The smoke test above is enough to verify that the build produced an importable module. Copy the built `.pyd` into the QO package only when you want notebooks in `QO-measurements` to use this newly built binary:

```bat
copy /Y build\windows-qom-ninja\AverageField.cp313-win_amd64.pyd C:\Users\Qop\QO-measurements\lib2\quantumOptics\
```

Then test import from the QO repository:

```bat
cd C:\Users\Qop\QO-measurements
python -c "import os; os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64'); import lib2.quantumOptics.AverageField as af; print(af); print([x for x in dir(af.AverageFieldMeasurer) if not x.startswith('_')])"
```

## Notes

- Use `digitizer_delay=90` for current hardware smoke tests. `digitizer_delay=0` currently triggers a `Spectrum_m4x.py` zero-sample-rate path before setup.
- CUDA DLLs are in `%CUDA_PATH%\bin\x64`, not `%CUDA_PATH%\bin`.
- `CMAKE_CUDA_ARCHITECTURES` is set to `native` in the preset for the RTX 5090. Override it in CMake settings only if native detection fails.
