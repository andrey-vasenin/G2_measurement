from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Import-smoke-test a built AverageField pybind11 module.")
    parser.add_argument("--build-dir", default="build/windows-qom-ninja", help="CMake build directory")
    parser.add_argument("--module", default=None, help="Explicit path to AverageField*.pyd")
    args = parser.parse_args()

    _add_dll_directories()
    module_path = Path(args.module) if args.module else _find_module(Path(args.build_dir))
    module_path = module_path.resolve()
    module = _load_module(module_path)

    methods = [name for name in dir(module.AverageFieldMeasurer) if not name.startswith("_")]
    print(f"Loaded: {module_path}")
    print(f"Module: {module}")
    print("AverageFieldMeasurer methods:")
    for method in methods:
        print(f"  {method}")


if __name__ == "__main__":
    main()
