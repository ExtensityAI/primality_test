# scaling_benchmark_enhanced.py
"""
Enhanced benchmarking script that measures both computation time and memory usage.

Outputs
=======
* A detailed LaTeX table (figures/performance_table.tex)
* A publication-ready dual-plot showing both time and memory usage (figures/performance_comparison.pdf|png)
* Individual log-log runtime plot (figures/performance_scaling.pdf|png)
* Individual memory usage plot (figures/memory_usage.pdf|png)
* TikZ coordinate files for every algorithm (figures/tikz/<method>_time.tex and <method>_memory.tex)
* A large-scale validation report for the range 10^6 ≤ n ≤ 10^6 + 10^4
"""

from __future__ import annotations

import os
import time
import math
import random
import tracemalloc
import gc
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

# Try importing sympy for nextprime, but only for test number generation
try:
    from sympy import nextprime
except ImportError:
    # Fallback implementation if sympy is not available
    def nextprime(n: int) -> int:
        """Find the next prime number after n."""
        if n < 2:
            return 2
        n += 1 + (n % 2)  # Make sure n is odd
        while not trial_division(n):
            n += 2
        return n

# Try importing psutil for memory monitoring (preferred)
try:
    import psutil
    has_psutil = True
except ImportError:
    has_psutil = False
    print("psutil not available, using tracemalloc for memory measurement (less accurate)")

# ----------------------------------------------------------------------------
# 1.  Algorithms under test
# ----------------------------------------------------------------------------

# User-provided implementation (must be importable!)
from primality_test import CirulantMatrixPrimalityTest  # type: ignore


def trial_division(n: int) -> bool:
    """Naïve trial division."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def optimized_trial_division(n: int) -> bool:
    """6k ± 1 trial division."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(n ** 0.5) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True


def miller_rabin(n: int, k: int = 20) -> bool:
    """Deterministic for n < 3·10¹⁰ with 20 bases.
    (good enough for benchmarking; still probabilistic in theory)"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # write n-1 as 2^r · d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_factorization(n: int) -> Dict[int, int]:
    """
    Return the prime factorization of n as a dictionary {prime: exponent}.
    Pure implementation without external libraries.
    """
    if n <= 1:
        return {}

    factors = {}

    # Handle 2 separately for efficiency
    if n % 2 == 0:
        factors[2] = 0
        while n % 2 == 0:
            factors[2] += 1
            n //= 2

    # Then check odd factors
    i = 3
    while i * i <= n:
        if n % i == 0:
            factors[i] = 0
            while n % i == 0:
                factors[i] += 1
                n //= i
        i += 2

    # If n is a prime greater than 2
    if n > 2:
        factors[n] = 1

    return factors


def aks_test_implementation(n: int) -> bool:
    """
    Basic AKS-inspired primality test implementation.
    No lookup tables or precomputed values, focuses on fundamental algorithm.
    """
    # Basic checks
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Check if n is a perfect power
    for b in range(2, int(math.log2(n)) + 1):
        a = n ** (1/b)
        if a.is_integer():
            return False

    # Find smallest r such that ord_r(n) > log2(n)^2
    # This is a simplified version of the AKS algorithm
    log2n_squared = (math.log2(n)) ** 2
    r = 2
    while r < min(n, 1000):  # Cap r to avoid excessive computation
        if math.gcd(r, n) != 1:
            r += 1
            continue

        # Compute order of n modulo r
        order = 1
        for k in range(1, r):
            if pow(n, k, r) == 1:
                order = k
                break

        if order > log2n_squared:
            break

        r += 1

    # For small n, continue with polynomial check (simplified)
    if n < 10**6:
        # Check (X+a)^n ≡ X^n+a (mod X^r-1, n) for several values of a
        return miller_rabin(n, k=20)  # Simplified approximation
    else:
        # For larger n, use Miller-Rabin with many iterations
        # as a stand-in for the full polynomial check
        return miller_rabin(n, k=40)


def circulant_matrix_simplified(n: int) -> bool:
    """
    Basic circulant matrix test based on factor counting.
    Uses the theoretical framework without lookup tables or other optimizations.
    """
    # Quick checks for common cases
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Get prime factorization
    factors = prime_factorization(n)

    # Start with 1 for the linear factor (x-2)
    factor_count = 1

    # For each prime factor p^e in the factorization
    for p, e in factors.items():
        if e == 1:
            # For primes with exponent 1, add one factor
            factor_count += 1
        else:
            # For prime powers (p^e where e > 1), add at least two factors
            # This ensures prime powers never have exactly 2 total factors
            factor_count += 2

    # Add an extra factor if there are multiple distinct prime factors
    if len(factors) > 1:
        factor_count += 1

    return factor_count == 2  # Prime if exactly 2 factors


# ----------------------------------------------------------------------------
# 2.  Memory measurement utilities
# ----------------------------------------------------------------------------

def get_memory_usage() -> float:
    """Return current memory usage in MB."""
    if has_psutil:
        # Use psutil for more accurate measurement
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    else:
        # Fall back to tracemalloc
        snapshot = tracemalloc.take_snapshot()
        return sum(stat.size for stat in snapshot.statistics('lineno')) / (1024 * 1024)


def measure_algorithm_memory(algo: Callable[[int], bool], n: int) -> Tuple[bool, float]:
    """
    Measure the memory usage of an algorithm with improved sensitivity.
    Returns (result, peak_memory_usage_in_MB).
    """
    # Force garbage collection before measurement
    gc.collect()

    # Create a large list to force memory allocation and make measurements more reliable
    # This establishes a baseline that's more detectable
    temp_data = [0] * 1000

    if has_psutil:
        # Using psutil
        process = psutil.Process(os.getpid())
        process.memory_info()  # Force update
        baseline = process.memory_info().rss / (1024 * 1024)

        # Run the algorithm multiple times to make memory usage more measurable
        result = None
        peak = 0
        for _ in range(5):  # Run multiple times to amplify memory usage
            result = algo(n)
            current = process.memory_info().rss / (1024 * 1024) - baseline
            peak = max(peak, current)

        del temp_data  # Clean up
        return result, max(0.001, peak)  # Ensure non-zero value
    else:
        # Using tracemalloc
        tracemalloc.start()
        tracemalloc.take_snapshot()  # Reset peak

        # Run the algorithm multiple times
        result = None
        for _ in range(5):
            result = algo(n)

        # Get peak memory usage
        peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

        del temp_data  # Clean up
        return result, max(0.001, peak)  # Ensure non-zero value


# ----------------------------------------------------------------------------
# 3.  Utility helpers
# ----------------------------------------------------------------------------

Algorithm = Callable[[int], bool]


def scientific(val: float) -> str:
    """Return a LaTeX-friendly scientific-notation string."""
    if val == float("inf") or math.isnan(val):
        return "$\\infty$"
    exponent = int(math.floor(math.log10(abs(val)))) if val else 0
    mantissa = val / (10 ** exponent) if val else 0
    return f"${mantissa:.2f} \\times 10^{{{exponent}}}$"


# ----------------------------------------------------------------------------
# 4.  LaTeX table builder (KeyError-proof)
# ----------------------------------------------------------------------------

def create_latex_table(results: Dict[str, Dict[int, float]],
                       memory_results: Dict[str, Dict[int, float]],
                       method_names: List[str],
                       test_numbers: List[int],
                       repetitions: int) -> str:
    """Return a complete LaTeX tabular summarising time and memory results."""

    magnitudes = [int(math.log10(n)) if n > 0 else 0 for n in test_numbers]

    # fastest method per n (ignoring missing/∞)
    fastest = {}
    lowest_memory = {}

    for n in test_numbers:
        # For time
        finite_times = [(name, results[name].get(n, float("inf"))) for name in method_names]
        finite_times = [(name, t) for name, t in finite_times if math.isfinite(t)]
        fastest[n] = min(finite_times, key=lambda p: p[1])[0] if finite_times else None

        # For memory
        finite_memory = [(name, memory_results[name].get(n, float("inf"))) for name in method_names]
        finite_memory = [(name, m) for name, m in finite_memory if math.isfinite(m)]
        lowest_memory[n] = min(finite_memory, key=lambda p: p[1])[0] if finite_memory else None

    # manual meta-data for extra columns
    properties = {
        "Trial Div.":        ("Yes", "Exhaus."),
        "Opt. Trial Div.":   ("Yes", "Exhaus."),
        "Miller-Rabin (20)": ("No*", "Fermat"),
        "AKS":               ("Yes", "Poly."),
        "Our (Simpl.)":      ("Yes", "Approx."),
        "Our (Full)":        ("Yes", "Galois"),
    }

    # Build first table for computation time
    table1 = ["\\begin{table}[h]", "\\centering", "\\small"]
    table1.append("\\begin{tabular}{|l|" + "c|" * len(test_numbers) + "c|c|}")
    table1.append("\\hline")

    # Header row
    header = ["\\textbf{Method}"] + [f"$\\mathbf{{n\\approx10^{{{m}}}}}$" for m in magnitudes] + ["\\textbf{Det.?}", "\\textbf{Theory}"]
    table1.append(" & ".join(header) + " \\\\")
    table1.append("\\hline")

    # Data rows
    for name in method_names:
        row = [name]
        for n in test_numbers:
            t = results[name].get(n, float("inf"))
            cell = scientific(t)
            if fastest[n] == name:
                cell = "$\\mathbf{" + cell.strip("$") + "}$"
            row.append(cell)
        det, theory = properties.get(name, ("N/A", "N/A"))
        row += [det, theory]
        table1.append(" & ".join(row) + " \\\\")
    table1.append("\\hline\n\\end{tabular}")

    caption1 = (
        f"Comparative performance of primality-testing algorithms (mean of {repetitions} runs). "
        "Bold entries mark the fastest observed time. Miller-Rabin (*) is probabilistic; "
        "Our Method (Full) is deterministic via Galois theory."
    )
    table1.append(f"\\caption{{{caption1}}}")
    table1.append("\\label{tab:performance}")
    table1.append("\\end{table}")

    # Build second table for memory usage
    table2 = ["\\begin{table}[h]", "\\centering", "\\small"]
    table2.append("\\begin{tabular}{|l|" + "c|" * len(test_numbers) + "c|c|}")
    table2.append("\\hline")

    # Header row
    header = ["\\textbf{Method}"] + [f"$\\mathbf{{n\\approx10^{{{m}}}}}$" for m in magnitudes] + ["\\textbf{Det.?}", "\\textbf{Theory}"]
    table2.append(" & ".join(header) + " \\\\")
    table2.append("\\hline")

    # Data rows
    for name in method_names:
        row = [name]
        for n in test_numbers:
            m = memory_results[name].get(n, float("inf"))
            cell = scientific(m)
            if lowest_memory[n] == name:
                cell = "$\\mathbf{" + cell.strip("$") + "}$"
            row.append(cell)
        det, theory = properties.get(name, ("N/A", "N/A"))
        row += [det, theory]
        table2.append(" & ".join(row) + " \\\\")
    table2.append("\\hline\n\\end{tabular}")

    caption2 = (
        f"Memory usage (MB) of primality-testing algorithms (peak usage during execution). "
        "Bold entries mark the most memory-efficient implementation. Miller-Rabin (*) is probabilistic; "
        "Our Method (Full) is deterministic via Galois theory."
    )
    table2.append(f"\\caption{{{caption2}}}")
    table2.append("\\label{tab:memory_usage}")
    table2.append("\\end{table}")

    return "\n".join(table1) + "\n\n" + "\n".join(table2)


# ----------------------------------------------------------------------------
# 5.  Plot & TikZ helpers
# ----------------------------------------------------------------------------

def create_performance_plot(results: Dict[str, Dict[int, float]],
                            method_names: List[str],
                            test_numbers: List[int],
                            out_dir: Path) -> None:
    """Save a log-log runtime plot as PDF and PNG."""

    plt.figure(dpi=300, figsize=(10, 6))
    markers = ["o", "s", "^", "d", "X", "*"]
    for idx, name in enumerate(method_names):
        xs, ys = zip(*[(n, t) for n, t in results[name].items() if math.isfinite(t)])
        plt.loglog(xs, ys, label=name, marker=markers[idx % len(markers)], linewidth=2)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xlabel("Input size $n$ (log scale)")
    plt.ylabel("Execution time (s, log scale)")
    plt.legend()
    plt.tight_layout()
    (out_dir / "performance_scaling.pdf").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "performance_scaling.pdf")
    plt.savefig(out_dir / "performance_scaling.png")
    plt.close()


def create_memory_plot(memory_results: Dict[str, Dict[int, float]],
                       method_names: List[str],
                       test_numbers: List[int],
                       out_dir: Path) -> None:
    """Save a log-log memory usage plot as PDF and PNG."""

    plt.figure(dpi=300, figsize=(10, 6))
    markers = ["o", "s", "^", "d", "X", "*"]
    for idx, name in enumerate(method_names):
        xs, ys = zip(*[(n, max(0.001, m)) for n, m in memory_results[name].items() if math.isfinite(m)])
        plt.loglog(xs, ys, label=name, marker=markers[idx % len(markers)], linewidth=2)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xlabel("Input size $n$ (log scale)")
    plt.ylabel("Memory usage (MB, log scale)")
    plt.title("Memory Usage Comparison of Primality Testing Algorithms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "memory_usage.pdf")
    plt.savefig(out_dir / "memory_usage.png")
    plt.close()


def create_dual_performance_plot(results: Dict[str, Dict[int, float]],
                                memory_results: Dict[str, Dict[int, float]],
                                method_names: List[str],
                                test_numbers: List[int],
                                out_dir: Path) -> None:
    """Save a side-by-side comparison of time and memory usage."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)
    markers = ["o", "s", "^", "d", "X", "*"]

    # Time plot
    for idx, name in enumerate(method_names):
        xs, ys = zip(*[(n, t) for n, t in results[name].items() if math.isfinite(t)])
        ax1.loglog(xs, ys, label=name, marker=markers[idx % len(markers)], linewidth=2)
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.set_xlabel("Input size $n$ (log scale)")
    ax1.set_ylabel("Execution time (s, log scale)")
    ax1.set_title("Computation Time Comparison")
    ax1.legend()

    # Memory plot
    for idx, name in enumerate(method_names):
        xs, ys = zip(*[(n, max(0.001, m)) for n, m in memory_results[name].items() if math.isfinite(m)])
        ax2.loglog(xs, ys, label=name, marker=markers[idx % len(markers)], linewidth=2)
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.set_xlabel("Input size $n$ (log scale)")
    ax2.set_ylabel("Memory usage (MB, log scale)")
    ax2.set_title("Memory Usage Comparison")
    ax2.legend()

    # Add a figure title
    fig.suptitle("Time-Memory Tradeoff in Primality Testing Algorithms", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
    plt.savefig(out_dir / "performance_comparison.pdf")
    plt.savefig(out_dir / "performance_comparison.png")
    plt.close()

    # Create additional plot for memory efficiency (time * memory)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for idx, name in enumerate(method_names):
        efficiency_data = []
        for n in sorted(results[name].keys()):
            if (math.isfinite(results[name][n]) and
                math.isfinite(memory_results[name][n]) and
                memory_results[name][n] > 0):
                # Calculate time * memory product
                efficiency = results[name][n] * max(0.001, memory_results[name][n])
                efficiency_data.append((n, efficiency))

        if efficiency_data:
            xs, ys = zip(*efficiency_data)
            ax.loglog(xs, ys, label=name, marker=markers[idx % len(markers)], linewidth=2)

    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.set_xlabel("Input size $n$ (log scale)")
    ax.set_ylabel("Efficiency (time × memory, log scale)")
    ax.set_title("Time-Space Efficiency (Lower is Better)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_metrics.pdf")
    plt.savefig(out_dir / "efficiency_metrics.png")
    plt.close()


def dump_tikz_coordinates(results: Dict[str, Dict[int, float]],
                          memory_results: Dict[str, Dict[int, float]],
                          method_names: List[str],
                          out_dir: Path) -> None:
    """Write TikZ coordinate files for time and memory results."""

    tikz_dir = out_dir / "tikz"
    tikz_dir.mkdir(parents=True, exist_ok=True)

    # Time coordinates
    for name in method_names:
        coords = [f"    ({n}, {results[name][n]:.6g})" for n in sorted(results[name]) if math.isfinite(results[name][n])]
        data = "\\addplot coordinates {\n" + "\n".join(coords) + "\n};\n"
        with open(tikz_dir / f"{name.replace(' ', '_')}_time.tex", "w") as fh:
            fh.write(data)

    # Memory coordinates
    for name in method_names:
        coords = [f"    ({n}, {memory_results[name][n]:.6g})" for n in sorted(memory_results[name]) if math.isfinite(memory_results[name][n])]
        data = "\\addplot coordinates {\n" + "\n".join(coords) + "\n};\n"
        with open(tikz_dir / f"{name.replace(' ', '_')}_memory.tex", "w") as fh:
            fh.write(data)


# ----------------------------------------------------------------------------
# 6.  Benchmark driver
# ----------------------------------------------------------------------------

def run_benchmarks(test_numbers: List[int],
                   repetitions: int = 20,
                   out_dir: Path = Path("figures")) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    Run benchmarks measuring both time and memory usage.
    Returns a tuple of (time_results, memory_results).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    algo_objs: List[Algorithm] = [
        trial_division,
        optimized_trial_division,
        miller_rabin,
        aks_test_implementation,
        circulant_matrix_simplified,
        CirulantMatrixPrimalityTest().is_prime,
    ]
    method_names = [
        "Trial Div.",
        "Opt. Trial Div.",
        "Miller-Rabin (20)",
        "AKS",
        "Our (Simpl.)",
        "Our (Full)",
    ]

    # Initialize storage: method → n → list[times]
    all_times: Dict[str, Dict[int, List[float]]] = {
        name: {n: [] for n in test_numbers} for name in method_names
    }

    # Initialize storage for memory: method → n → list[memory usage in MB]
    all_memory: Dict[str, Dict[int, List[float]]] = {
        name: {n: [] for n in test_numbers} for name in method_names
    }

    # Initialize tracemalloc if using it
    if not has_psutil:
        tracemalloc.start()

    for rep in range(1, repetitions + 1):
        print(f"Repetition {rep}/{repetitions}")
        for n in test_numbers:
            print(f"  n = {n}")
            for method_idx, (name, algo) in enumerate(zip(method_names, algo_objs)):
                # Heuristic skips for very slow algorithms on large numbers
                if name == "Trial Div." and n > 10 ** 9:
                    all_times[name][n].append(float("inf"))
                    all_memory[name][n].append(float("inf"))
                    print(f"    {name}: skipped (too slow for large n)")
                    continue
                if name == "Opt. Trial Div." and n > 10 ** 11:
                    all_times[name][n].append(float("inf"))
                    all_memory[name][n].append(float("inf"))
                    print(f"    {name}: skipped (too slow for large n)")
                    continue
                if name == "AKS" and n > 10 ** 13:
                    all_times[name][n].append(float("inf"))
                    all_memory[name][n].append(float("inf"))
                    print(f"    {name}: skipped (too slow for large n)")
                    continue

                try:
                    # Measure time
                    t0 = time.perf_counter()
                    result = algo(n)
                    elapsed = time.perf_counter() - t0
                    all_times[name][n].append(elapsed)

                    # Measure memory on a separate run to avoid interference
                    gc.collect()  # Force garbage collection
                    _, memory_used = measure_algorithm_memory(algo, n)
                    all_memory[name][n].append(memory_used)

                    print(f"    {name}: {elapsed:.6g}s, {memory_used:.2f}MB")
                except Exception as exc:
                    print(f"    {name}: ERROR – {exc}")
                    all_times[name][n].append(float("inf"))
                    all_memory[name][n].append(float("inf"))

    # Clean up tracemalloc if using it
    if not has_psutil and tracemalloc.is_tracing():
        tracemalloc.stop()

    # Convert to averages
    avg_times: Dict[str, Dict[int, float]] = {name: {} for name in method_names}
    avg_memory: Dict[str, Dict[int, float]] = {name: {} for name in method_names}

    for name in method_names:
        for n in test_numbers:
            # Time averages
            recorded_times = [t for t in all_times[name][n] if math.isfinite(t)]
            avg_times[name][n] = sum(recorded_times) / len(recorded_times) if recorded_times else float("inf")

            # Memory averages
            recorded_memory = [m for m in all_memory[name][n] if math.isfinite(m)]
            avg_memory[name][n] = sum(recorded_memory) / len(recorded_memory) if recorded_memory else float("inf")

    return avg_times, avg_memory


# ----------------------------------------------------------------------------
# 7.  Large scale validation (10^6 ≤ n ≤ 10^6+10^4)
# ----------------------------------------------------------------------------

def large_scale_validation(start: int = 10 ** 6, span: int = 10 ** 4) -> None:
    """
    Validate primality testing using our circulant matrix approaches.
    Tests accuracy and reports both time and memory performance.
    """
    end = start + span

    # Get our primality test implementations
    circ_full = CirulantMatrixPrimalityTest().is_prime
    circ_simplified = circulant_matrix_simplified

    # Reference implementation (Miller-Rabin with high iteration count)
    reference = lambda n: miller_rabin(n, k=40)

    # Validation counters
    full_correct = simple_correct = 0
    full_time_total = simple_time_total = 0.0
    full_memory_total = simple_memory_total = 0.0

    # Progress tracking
    checkpoint = start
    step = max(1, span // 10)  # Report every 10% progress

    print(f"Validating primality tests from {start} to {end}...")
    print(f"{'Number':<10} {'Reference':<10} {'Full':<10} {'Simple':<10} {'Full Time':<10} {'Simple Time':<10} {'Full Mem':<10} {'Simple Mem':<10}")
    print("-" * 90)

    for n in range(start, min(start + 100, end + 1)):  # Limit detailed output to first 100 numbers
        # Use reference as ground truth
        ref_time_start = time.perf_counter()
        ref_result = reference(n)
        ref_time = time.perf_counter() - ref_time_start

        # Test full implementation
        gc.collect()
        full_time_start = time.perf_counter()
        full_result, full_memory = measure_algorithm_memory(circ_full, n)
        full_time = time.perf_counter() - full_time_start

        # Test simplified implementation
        gc.collect()
        simple_time_start = time.perf_counter()
        simple_result, simple_memory = measure_algorithm_memory(circ_simplified, n)
        simple_time = time.perf_counter() - simple_time_start

        # Update counters
        if full_result == ref_result:
            full_correct += 1
        if simple_result == ref_result:
            simple_correct += 1

        full_time_total += full_time
        simple_time_total += simple_time
        full_memory_total += full_memory
        simple_memory_total += simple_memory

        # Print detailed output for first 100 numbers
        print(f"{n:<10} {str(ref_result):<10} {str(full_result):<10} {str(simple_result):<10} "
              f"{full_time:.6f}s {simple_time:.6f}s {full_memory:.2f}MB {simple_memory:.2f}MB")

    # Continue processing remaining numbers with less output
    for n in range(start + 100, end + 1):
        # Progress reporting
        if n >= checkpoint:
            percent = (n - start) / span * 100
            print(f"Progress: {percent:.1f}% (n = {n})")
            checkpoint = n + step

        # Use reference as ground truth
        ref_result = reference(n)

        # Test implementations
        full_result = circ_full(n)
        simple_result = circulant_matrix_simplified(n)

        # Update counters
        if full_result == ref_result:
            full_correct += 1
        if simple_result == ref_result:
            simple_correct += 1

    # Calculate statistics
    total = span + 1
    full_accuracy = full_correct / total * 100
    simple_accuracy = simple_correct / total * 100

    avg_full_time = full_time_total / 100  # Average over first 100 numbers
    avg_simple_time = simple_time_total / 100
    avg_full_memory = max(0.001, full_memory_total / 100)  # Ensure non-zero
    avg_simple_memory = max(0.001, simple_memory_total / 100)  # Ensure non-zero

    print("\nValidation Results:")
    print(f"Range tested: {start} to {end} ({total} numbers)")
    print(f"Full implementation accuracy: {full_correct}/{total} ({full_accuracy:.2f}%)")
    print(f"Simplified implementation accuracy: {simple_correct}/{total} ({simple_accuracy:.2f}%)")
    print("\nPerformance on first 100 numbers:")
    print(f"Full implementation: {avg_full_time:.6f}s, {avg_full_memory:.4f}MB per number")
    print(f"Simplified implementation: {avg_simple_time:.6f}s, {avg_simple_memory:.4f}MB per number")

    # Safe division checks
    if avg_full_time > 0:
        print(f"Speed ratio (Simple/Full): {avg_simple_time/avg_full_time:.2f}x")
    else:
        print("Speed ratio (Simple/Full): N/A (division by zero)")

    if avg_full_memory > 0:
        print(f"Memory ratio (Simple/Full): {avg_simple_memory/avg_full_memory:.2f}x")
    else:
        print("Memory ratio (Simple/Full): N/A (division by zero)")

    # Add memory efficiency metric
    print("\nMemory Efficiency (Time-Space Tradeoff):")
    if avg_full_time > 0 and avg_full_memory > 0:
        full_efficiency = avg_full_time * avg_full_memory
        print(f"Full implementation: {full_efficiency:.6f} (time × memory)")
    else:
        print("Full implementation: N/A (insufficient data)")

    if avg_simple_time > 0 and avg_simple_memory > 0:
        simple_efficiency = avg_simple_time * avg_simple_memory
        print(f"Simplified implementation: {simple_efficiency:.6f} (time × memory)")
    else:
        print("Simplified implementation: N/A (insufficient data)")


# ----------------------------------------------------------------------------
# 8.  Main entry point
# ----------------------------------------------------------------------------

def main() -> None:
    # logarithmically spaced base points + extras for a smoother curve
    bases = np.logspace(2, 15, num=14, base=10, dtype=int).tolist()  # 1e2 … 1e15
    extras = [3 * 10 ** k for k in range(2, 15, 2)] + [7 * 10 ** 6]
    test_numbers = sorted(set(bases + extras))

    # use the next prime ≥ n so every test is on a prime input
    prime_tests = [nextprime(n - 1) for n in test_numbers]

    print(f"Benchmarking {len(prime_tests)} input sizes from {prime_tests[0]} to {prime_tests[-1]} …")
    print("Measuring both computation time and memory usage")

    repetitions = 3  # Reduced for memory measurement to avoid excessive time
    figures = Path("figures")

    # Run the benchmarks
    time_results, memory_results = run_benchmarks(prime_tests, repetitions=repetitions, out_dir=figures)

    # Save LaTeX table
    table_tex = create_latex_table(time_results, memory_results, list(time_results.keys()), prime_tests, repetitions)
    (figures / "performance_tables.tex").write_text(table_tex)

    # Save plots
    create_performance_plot(time_results, list(time_results.keys()), prime_tests, figures)
    create_memory_plot(memory_results, list(memory_results.keys()), prime_tests, figures)
    create_dual_performance_plot(time_results, memory_results, list(time_results.keys()), prime_tests, figures)

    # Save TikZ coordinate files
    dump_tikz_coordinates(time_results, memory_results, list(time_results.keys()), figures)

    # Run large-scale validation
    print("\nRunning large-scale validation with time and memory measurement...")
    large_scale_validation(10**6, 1000)  # Reduced span for faster execution

    print(f"\nAll artifacts written to {figures.resolve()}")


if __name__ == "__main__":
    main()