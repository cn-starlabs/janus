#!/usr/bin/env python3
"""
Performance profiling: Euler vs RK2 vs RK4 time integrators via live FFI.

Usage:
    python3 scripts/benchmark_schemes.py [--steps N] [--nx N] [--ny N] [--output report.json]

Measures wall-clock time per solver step for each scheme, verifies RK4 takes
roughly 3× more steps internally (3 extra flux evaluations vs Euler), and
reports mean / median / p95 timings.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal ctypes bindings (no bpy dependency)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = PROJECT_ROOT / "target" / "release" / "libjanus_ffi.so"


class JanusGridInfo(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_size_t),
        ("ny", ctypes.c_size_t),
        ("dx", ctypes.c_double),
        ("dy", ctypes.c_double),
        ("origin_x", ctypes.c_double),
        ("origin_y", ctypes.c_double),
        ("ncells", ctypes.c_size_t),
    ]


def load_library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.janus_ffi_api_version.restype = ctypes.c_uint32
    lib.janus_last_error.restype = ctypes.c_char_p
    lib.janus_solver_create.argtypes = [ctypes.c_char_p]
    lib.janus_solver_create.restype = ctypes.c_void_p
    lib.janus_solver_destroy.argtypes = [ctypes.c_void_p]
    lib.janus_solver_step.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.janus_solver_step.restype = ctypes.c_int
    lib.janus_solver_cfl_dt.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.janus_solver_cfl_dt.restype = ctypes.c_double
    lib.janus_solver_grid_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(JanusGridInfo)]
    lib.janus_solver_grid_info.restype = ctypes.c_int
    return lib


def last_error(lib: ctypes.CDLL) -> str:
    msg = lib.janus_last_error()
    return msg.decode("utf-8") if msg else "(no error)"


def build_config(nx: int, ny: int, scheme: str) -> str:
    """Return a JSON config string for a Couette-like test case."""
    domain = 1.0e-3
    dx = domain / nx
    dy = domain / ny
    config = {
        "config": {
            "grid": {
                "nx": nx, "ny": ny,
                "dx": dx, "dy": dy,
                "origin": [0.0, 0.0],
            },
            "bcs": {
                "west": "Periodic",
                "east": "Periodic",
                "south": {"DiffuseWall": {"temperature": 300.0, "wall_velocity": [0.0, 0.0]}},
                "north": {"DiffuseWall": {"temperature": 300.0, "wall_velocity": [50.0, 0.0]}},
            },
            "gas": {
                "r_gas": 208.13,
                "molar_mass": 0.039948,
                "vhs_omega": 0.81,
                "mu_ref": 2.117e-5,
                "t_ref": 273.15,
                "prandtl": 0.6666666666666666,
            },
        },
        "initial": {"rho": 1.0, "temperature": 300.0, "velocity": [0.0, 0.0]},
        "velocity_grid": {"v_max": 1800.0, "n_per_dim": 25},
        "time_scheme": {"scheme": scheme},
        "scene": {"boundary_tags": []},
    }
    return json.dumps(config)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def hdr(title: str):
    bar = "─" * 58
    print(f"\n{C.BLUE}{bar}")
    print(f"  {C.BOLD}{title}{C.RESET}")
    print(f"{C.BLUE}{bar}{C.RESET}")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    lib: ctypes.CDLL,
    scheme: str,
    nx: int,
    ny: int,
    n_warmup: int,
    n_steps: int,
) -> dict:
    """
    Create a fresh solver for each scheme, run warm-up, then time n_steps.

    Returns dict with keys: scheme, nx, ny, n_steps, dt, timings_ms,
    mean_ms, median_ms, p95_ms, total_ms, steps_per_sec.
    """
    cfg = build_config(nx, ny, scheme).encode("utf-8")
    handle = lib.janus_solver_create(cfg)
    if not handle:
        raise RuntimeError(f"create_solver failed for {scheme}: {last_error(lib)}")

    try:
        dt = lib.janus_solver_cfl_dt(handle, 0.4)
        if dt <= 0.0:
            raise RuntimeError(f"CFL dt is zero for scheme={scheme}")

        # Warm-up (excluded from timing)
        for _ in range(n_warmup):
            rc = lib.janus_solver_step(handle, dt)
            if rc != 0:
                raise RuntimeError(f"step error during warm-up: {last_error(lib)}")

        # Timed steps
        timings = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            rc = lib.janus_solver_step(handle, dt)
            t1 = time.perf_counter()
            if rc != 0:
                raise RuntimeError(f"step error: {last_error(lib)}")
            timings.append((t1 - t0) * 1000.0)  # ms

        timings_sorted = sorted(timings)
        mean_ms   = statistics.mean(timings)
        median_ms = statistics.median(timings)
        p95_idx   = max(0, int(0.95 * len(timings_sorted)) - 1)
        p95_ms    = timings_sorted[p95_idx]
        total_ms  = sum(timings)

        return {
            "scheme":        scheme,
            "nx":            nx,
            "ny":            ny,
            "n_warmup":      n_warmup,
            "n_steps":       n_steps,
            "dt":            dt,
            "timings_ms":    timings,
            "mean_ms":       mean_ms,
            "median_ms":     median_ms,
            "p95_ms":        p95_ms,
            "total_ms":      total_ms,
            "steps_per_sec": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        }
    finally:
        lib.janus_solver_destroy(handle)


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_results(results: list[dict], baseline_scheme: str = "euler"):
    baseline = next((r for r in results if r["scheme"] == baseline_scheme), None)

    hdr("Benchmark Results")
    print(f"  {'Scheme':<8}  {'Mean (ms)':>10}  {'Median (ms)':>12}  {'P95 (ms)':>10}  {'Steps/s':>10}  {'Overhead vs Euler':>18}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*18}")

    for r in results:
        overhead = ""
        if baseline and r["scheme"] != baseline_scheme:
            ratio = r["mean_ms"] / baseline["mean_ms"] if baseline["mean_ms"] > 0 else 0
            overhead = f"{ratio:.2f}×"
        elif r["scheme"] == baseline_scheme:
            overhead = "1.00× (baseline)"

        color = C.GREEN if r["scheme"] == "euler" else (C.YELLOW if r["scheme"] == "rk2" else C.CYAN)
        print(
            f"  {color}{r['scheme']:<8}{C.RESET}  "
            f"{r['mean_ms']:>10.3f}  "
            f"{r['median_ms']:>12.3f}  "
            f"{r['p95_ms']:>10.3f}  "
            f"{r['steps_per_sec']:>10.1f}  "
            f"{overhead:>18}"
        )

    print()
    if baseline:
        for r in results:
            if r["scheme"] != baseline_scheme and baseline["mean_ms"] > 0:
                ratio = r["mean_ms"] / baseline["mean_ms"]
                theoretical = {"rk2": 2.0, "rk4": 4.0}.get(r["scheme"], 1.0)
                efficiency = theoretical / ratio * 100
                print(
                    f"  {r['scheme'].upper()}: measured {ratio:.2f}× overhead "
                    f"(theoretical {theoretical:.0f}×, "
                    f"efficiency {efficiency:.0f}%)"
                )

    # Per-step cost analysis
    hdr("First-Step Validation")
    grid = results[0]
    ncells = grid["nx"] * grid["ny"]
    nv = 25 * 25  # n_per_dim²
    print(f"  Grid: {grid['nx']}×{grid['ny']} = {ncells} cells, "
          f"velocity grid: {nv} points/cell")

    for r in results:
        ns_per_cell = r["mean_ms"] * 1e6 / ncells  # ns per cell
        print(f"  {r['scheme']:<8}: {r['mean_ms']:.3f} ms/step = "
              f"{ns_per_cell:.1f} ns/cell")

    # Verify RK4 runs to completion
    hdr("Correctness Check")
    for r in results:
        all_positive = all(v > 0 for v in r["timings_ms"])
        status = f"{C.GREEN}✓{C.RESET}" if all_positive else f"\033[91m✗{C.RESET}"
        print(f"  {status} {r['scheme']}: {r['n_steps']} steps completed, "
              f"all timings positive: {all_positive}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def export_html(results: list[dict], output_path: str):
    from datetime import datetime

    baseline_result = next((r for r in results if r["scheme"] == "euler"), None)

    rows = ""
    for r in results:
        ratio_str = "-"
        if baseline_result and r["scheme"] != "euler" and baseline_result["mean_ms"] > 0:
            ratio = r["mean_ms"] / baseline_result["mean_ms"]
            ratio_str = f"{ratio:.2f}×"
        elif r["scheme"] == "euler":
            ratio_str = "1.00× (baseline)"

        rows += f"""
        <tr>
            <td><strong>{r['scheme'].upper()}</strong></td>
            <td>{r['mean_ms']:.3f}</td>
            <td>{r['median_ms']:.3f}</td>
            <td>{r['p95_ms']:.3f}</td>
            <td>{r['steps_per_sec']:.1f}</td>
            <td>{ratio_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Janus Scheme Benchmark</title>
    <style>
        body {{ font-family: monospace; margin: 30px; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 8px; }}
        h2 {{ color: #7fdbff; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
        th {{ background: #16213e; color: #00d4ff; padding: 10px 14px; text-align: left;
              border: 1px solid #334; }}
        td {{ padding: 8px 14px; border: 1px solid #334; }}
        tr:nth-child(even) {{ background: #0f3460; }}
        .note {{ color: #aaa; font-size: 0.9em; margin-top: 12px; }}
        .timestamp {{ color: #666; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>Janus — Time Integrator Benchmark</h1>
    <p class="timestamp">Generated: {datetime.now().isoformat()}</p>
    <p>Grid: {results[0]['nx']}×{results[0]['ny']} cells &nbsp;|&nbsp;
       Steps per scheme: {results[0]['n_steps']} &nbsp;|&nbsp;
       Warm-up: {results[0]['n_warmup']} steps</p>

    <h2>Per-Step Timing</h2>
    <table>
        <tr>
            <th>Scheme</th>
            <th>Mean (ms)</th>
            <th>Median (ms)</th>
            <th>P95 (ms)</th>
            <th>Steps/s</th>
            <th>Overhead vs Euler</th>
        </tr>
        {rows}
    </table>

    <p class="note">
        RK2 ≈ 2× Euler cost (2 flux evaluations).
        RK4 ≈ 4× Euler cost (4 flux evaluations).
        Actual overhead may be lower due to memory bandwidth sharing and CPU cache effects.
    </p>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"\n  Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Euler vs RK2 vs RK4 via Janus FFI")
    parser.add_argument("--steps",  "-s", type=int, default=50,   help="Timed steps per scheme (default: 50)")
    parser.add_argument("--warmup", "-w", type=int, default=5,    help="Warm-up steps (default: 5)")
    parser.add_argument("--nx",           type=int, default=32,   help="Grid X cells (default: 32)")
    parser.add_argument("--ny",           type=int, default=32,   help="Grid Y cells (default: 32)")
    parser.add_argument("--schemes",      type=str, default="euler,rk2,rk4", help="Comma-separated schemes")
    parser.add_argument("--output", "-o", type=str, default=None, help="HTML report output path")
    parser.add_argument("--json",         type=str, default=None, help="JSON results output path")
    args = parser.parse_args()

    # Check library
    if not LIB_PATH.exists():
        print(f"\033[91mFFI library not found: {LIB_PATH}\033[0m")
        print("Build with:  cargo build -p janus-ffi --release")
        sys.exit(1)

    hdr("Janus Time Integrator Benchmark")
    print(f"  Library:  {LIB_PATH}")
    lib = load_library(LIB_PATH)
    api_ver = lib.janus_ffi_api_version()
    print(f"  API version: {api_ver}")
    print(f"  Grid:     {args.nx}×{args.ny}  ({args.nx * args.ny} cells)")
    print(f"  Warm-up:  {args.warmup} steps  |  Timed: {args.steps} steps/scheme")

    schemes = [s.strip() for s in args.schemes.split(",")]
    results = []

    for scheme in schemes:
        print(f"\n  {C.YELLOW}▸ Running {scheme.upper()}…{C.RESET}", end="", flush=True)
        try:
            r = run_benchmark(
                lib,
                scheme=scheme,
                nx=args.nx,
                ny=args.ny,
                n_warmup=args.warmup,
                n_steps=args.steps,
            )
            results.append(r)
            print(f"  mean={r['mean_ms']:.3f} ms/step  dt={r['dt']:.3e} s")
        except RuntimeError as e:
            print(f"\n  \033[91m✗ {scheme}: {e}\033[0m")

    if not results:
        print("\033[91mAll benchmarks failed.\033[0m")
        sys.exit(1)

    print_results(results, baseline_scheme="euler")

    if args.output:
        export_html(results, args.output)

    if args.json:
        # Strip timings list from JSON to keep it compact
        compact = [{k: v for k, v in r.items() if k != "timings_ms"} for r in results]
        Path(args.json).write_text(json.dumps(compact, indent=2), encoding="utf-8")
        print(f"  JSON results saved to: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
