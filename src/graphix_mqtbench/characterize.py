from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from graphix_mqtbench import Benchmark, BenchmarkResult, OptimizationPass
from graphix_mqtbench._generated_benchmarks_enum import BenchmarkName

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from graphix.pattern import Pattern


def characterize_benchmark(benchmark: Benchmark, pretty: bool = True) -> pd.DataFrame:
    data: dict[str, str | int] = {}

    def add_data(name: str, pattern: Pattern) -> None:
        data[f"{name}-max_space"] = pattern.max_space()
        data[f"{name}-n_commands"] = len(pattern)

    data["benchmark"] = benchmark.name.value
    data["nqubits"] = benchmark.nqubits

    circuit = benchmark.to_circuit()
    data["n_gates"] = len(circuit.instruction)

    pattern = circuit.transpile().pattern

    add_data("transp", pattern)

    # If we iterate over all passes, we would be doing P twice.

    optim = OptimizationPass.M
    add_data(optim.name, optim.apply_optimization(pattern))

    name = ""
    for optim in (OptimizationPass.P, OptimizationPass.M):
        pattern = optim.apply_optimization(pattern)
        name += optim.name
        add_data(name, pattern)

    df = pd.DataFrame([data])

    if pretty:
        return prettify_benchmark_df(df)

    return df


def characterize_all_benchmarks(nqubits: int) -> pd.DataFrame:

    empty_dict = {
        f"{optim.name}-{field}": np.nan for optim in OptimizationPass for field in ["max_space", "n_commands"]
    }

    rows = []
    for bench in BenchmarkName:
        try:
            df = Benchmark(bench, nqubits).characterize(pretty=False)
        except ValueError:
            df = pd.DataFrame(
                [
                    {
                        "benchmark": bench.name,
                        "nqubits": nqubits,
                        "n_gates": np.nan,
                        "transp-max_space": np.nan,
                        "transp-n_commands": np.nan,
                        **empty_dict,
                    }
                ]
            )

        rows.append(df)

    return prettify_benchmark_df(pd.concat(rows, ignore_index=True, sort=False))


def characterize_benchmarks(benchmarks: Sequence[Benchmark]) -> pd.DataFrame:
    return prettify_benchmark_df(
        pd.concat([benchmark.characterize(pretty=False) for benchmark in benchmarks], ignore_index=True, sort=False)
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
        optim = f"-{r.extra_info['optim']}" if r.extra_info['optim'] else ""
        df["Backend"] = r.extra_info['backend_name'] + optim

        rows.append(df)

    long_df = pd.concat(rows, ignore_index=True)

    # `swaplevel(axis=1)` places `Backend` above data.
    return long_df.pivot_table(index=["Benchmark", "# Qubits"], columns=["Backend"], aggfunc="first").swaplevel(axis=1)


def read_results(paths: Iterable[Path]) -> pd.DataFrame:
    results: list[BenchmarkResult] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        results.extend(BenchmarkResult.from_dict(data_dict) for data_dict in data['benchmarks'])

    return combine_benchmark_results(results)


def read_all_benchmarks() -> pd.DataFrame:
    folder = Path.cwd() / ".benchmarks/"
    return read_results(folder.rglob('*json'))
