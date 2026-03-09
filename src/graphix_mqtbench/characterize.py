from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from graphix_mqtbench import Benchmark, BenchmarkResult, OptimizationPass
from graphix_mqtbench._generated_benchmarks_enum import BenchmarkName
from graphix_mqtbench.benchmark import BenchmarkError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from graphix.pattern import Pattern
    from graphix.transpiler import Circuit


def characterize_benchmark(
    benchmark: Benchmark, optim_passes: Sequence[OptimizationPass] = (), pretty: bool = True
) -> pd.DataFrame:
    data: dict[str, str | int] = {}

    def compute_patterns(circuit: Circuit) -> dict[str, Pattern]:
        base_pattern = circuit.transpile().pattern
        patterns: dict[str, Pattern] = {"transp": base_pattern}
        cache: dict[str, Pattern] = {}

        for optim_pass in optim_passes:
            if optim_pass.name not in cache:
                name = ""
                pattern = base_pattern
                for op in optim_pass.name:
                    name += op
                    if name in cache:
                        pattern = cache[name]
                    else:
                        cache[name] = OptimizationPass[name].apply_optimization(pattern)

            patterns[optim_pass.name] = cache[optim_pass.name]

        return patterns

    data["benchmark"] = benchmark.name.value
    data["nqubits"] = benchmark.nqubits

    circuit = benchmark.to_circuit()
    data["n_gates"] = len(circuit.instruction)

    patterns = compute_patterns(circuit)

    for name, pattern in patterns.items():
        data[f"{name}-max_space"] = pattern.max_space()
        data[f"{name}-n_commands"] = len(pattern)

    df = pd.DataFrame([data])

    if pretty:
        return prettify_benchmark_df(df)

    return df


def characterize_or_empty(
    bench: BenchmarkName, nqubits: int, optim_passes: Sequence[OptimizationPass] = ()
) -> pd.DataFrame:
    try:
        return Benchmark(bench, nqubits).characterize(optim_passses=optim_passes, pretty=False)
    except BenchmarkError:
        return pd.DataFrame(
            [
                {
                    "benchmark": bench.name,
                    "nqubits": nqubits,
                }
            ]
        )


def characterize_all_benchmarks(nqubits: int, optim_passes: Sequence[OptimizationPass] = ()) -> pd.DataFrame:
    rows = [characterize_or_empty(bench, nqubits, optim_passes) for bench in BenchmarkName]
    df = pd.concat(rows, ignore_index=True, sort=False)
    return prettify_benchmark_df(df)


def characterize_benchmarks(
    benchmarks: Sequence[Benchmark], optim_passes: Sequence[OptimizationPass] = ()
) -> pd.DataFrame:
    return prettify_benchmark_df(
        pd.concat(
            [benchmark.characterize(optim_passses=optim_passes, pretty=False) for benchmark in benchmarks],
            ignore_index=True,
            sort=False,
        )
    )


def prettify_benchmark_df(df: pd.DataFrame) -> pd.DataFrame:
    new_columns = []

    mapping = {
        "transp-": "Transpilation",
        "P-": "Pauli presimulation",
        "M-": "Space minimization",
        "PM-": "Pauli presimulation + Space minimization",
    }

    for col in df.columns:
        if col != "benchmark":
            df[col] = df[col].astype("Int64")
        if col in {"benchmark", "nqubits", "n_gates"}:
            new_columns.append(("Circuit", col))
        else:
            for key, value in mapping.items():
                if col.startswith(key):
                    new_columns.append((value, col.replace(key, "")))

    df.columns = pd.MultiIndex.from_tuples(new_columns)
    return df.rename(
        columns={
            "n_commands": "# Commands",
            "max_space": "Max Space",
            "n_gates": "# Gates",
            "nqubits": "# Qubits",
            "benchmark": "Benchmark",
        },
        level=1,
    )


def combine_benchmark_results(results: Sequence[BenchmarkResult]) -> pd.DataFrame:
    rows = []

    for r in results:
        df = r.stats.copy()
        df["Benchmark"] = r.benchmark.name.value
        df["# Qubits"] = r.benchmark.nqubits
        optim = f"-{r.extra_info['optim']}" if r.extra_info["optim"] else ""
        df["Backend"] = r.extra_info["backend_name"] + optim

        rows.append(df)

    long_df = pd.concat(rows, ignore_index=True)

    # `swaplevel(axis=1)` places `Backend` above data.
    return long_df.pivot_table(index=["Benchmark", "# Qubits"], columns=["Backend"], aggfunc="first").swaplevel(axis=1)


def read_results(paths: Iterable[Path]) -> pd.DataFrame:
    results: list[BenchmarkResult] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        results.extend(BenchmarkResult.from_dict(data_dict) for data_dict in data["benchmarks"])

    return combine_benchmark_results(results)


def read_all_benchmarks() -> pd.DataFrame:
    folder = Path.cwd() / ".benchmarks/"
    return read_results(folder.rglob("*json"))
