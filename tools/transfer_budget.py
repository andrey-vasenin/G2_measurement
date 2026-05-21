from __future__ import annotations

import argparse
import sys


MIB = 1024 * 1024


def _mib(bytes_count: float) -> float:
    return bytes_count / MIB


def _raw_mib_s(segment_samples: int, channels: int, sample_bytes: int, period_ns: float) -> float:
    bytes_per_trigger = segment_samples * channels * sample_bytes
    return _mib(bytes_per_trigger * 1e9 / period_ns)


def _min_period_ns(segment_samples: int, channels: int, sample_bytes: int, fifo_mib_s: float) -> float:
    bytes_per_trigger = segment_samples * channels * sample_bytes
    return bytes_per_trigger / (fifo_mib_s * MIB) * 1e9


def _format_status(raw_mib_s: float, fifo_mib_s: float, safety_margin: float) -> str:
    safe_limit = fifo_mib_s * (1.0 - safety_margin)
    if raw_mib_s > fifo_mib_s:
        return "OVER"
    if raw_mib_s > safe_limit:
        return "NEAR"
    return "OK"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate Spectrum FIFO and host-to-GPU transfer budget for triggered acquisition settings."
    )
    parser.add_argument("--segment-samples", type=int, default=1248, help="Configured samples per segment")
    parser.add_argument("--n-seg", type=int, default=8192, help="Segments per notify/batch")
    parser.add_argument("--period-ns", type=float, default=1000.0, help="Trigger repetition period in ns")
    parser.add_argument("--channels", type=int, nargs="+", default=[2, 4], help="Physical digitizer channel counts")
    parser.add_argument("--sample-bytes", type=int, default=1, help="Bytes per physical channel sample")
    parser.add_argument(
        "--fifo-mib-s",
        type=float,
        default=2618.5,
        help="Measured sustained Spectrum FIFO read bandwidth in MiB/s",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.20,
        help="Fractional FIFO headroom required for OK status",
    )
    parser.add_argument(
        "--fail-on-over",
        action="store_true",
        help="Exit non-zero if any channel count exceeds measured FIFO bandwidth",
    )
    args = parser.parse_args()

    print("Spectrum transfer budget")
    print(f"  segment_samples: {args.segment_samples}")
    print(f"  n_seg:           {args.n_seg}")
    print(f"  period_ns:       {args.period_ns:g}")
    print(f"  sample_bytes:    {args.sample_bytes}")
    print(f"  fifo_mib_s:      {args.fifo_mib_s:g}")
    print(f"  safety_margin:   {args.safety_margin:.0%}")
    print()
    print(
        "channels  notify MiB  host buffer MiB  raw MiB/s  "
        "headroom %  min period ns  min safe ns  status"
    )
    print("-" * 94)

    has_over = False
    for channels in args.channels:
        notify_bytes = args.segment_samples * channels * args.sample_bytes * args.n_seg
        host_buffer_bytes = 4 * notify_bytes
        raw_mib_s = _raw_mib_s(args.segment_samples, channels, args.sample_bytes, args.period_ns)
        headroom = (args.fifo_mib_s / raw_mib_s - 1.0) * 100.0
        min_period = _min_period_ns(args.segment_samples, channels, args.sample_bytes, args.fifo_mib_s)
        min_safe_period = min_period / (1.0 - args.safety_margin)
        status = _format_status(raw_mib_s, args.fifo_mib_s, args.safety_margin)
        has_over = has_over or status == "OVER"

        print(
            f"{channels:8d}  "
            f"{_mib(notify_bytes):10.2f}  "
            f"{_mib(host_buffer_bytes):15.2f}  "
            f"{raw_mib_s:9.1f}  "
            f"{headroom:10.1f}  "
            f"{min_period:13.0f}  "
            f"{min_safe_period:11.0f}  "
            f"{status}"
        )

    if args.fail_on_over and has_over:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
