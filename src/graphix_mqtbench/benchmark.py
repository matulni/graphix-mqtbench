from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from mqt.bench import get_benchmark_indep

from graphix_mqtbench._generated_benchmarks_enum import BenchmarkName
from graphix_mqtbench.converter import qiskit_to_graphix_circuit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from graphix.pattern import Pattern
    from graphix.transpiler import Circuit


@dataclass
class Benchmark:
    name: BenchmarkName
    nqubits: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        try:
            get_benchmark_indep(benchmark=self.name.value, circuit_size=self.nqubits)
        except Exception as e:
            raise ValueError(f"{self.name.value} benchmark does not exist for {self.nqubits} qubits.") from e

    def to_circuit(self) -> Circuit:
        return qiskit_to_graphix_circuit(
            get_benchmark_indep(benchmark=self.name.value, circuit_size=self.nqubits),
        )

    def to_pattern(self, pauli_presimulate: bool = True, min_space: bool = True) -> Pattern:
        pattern = self.to_circuit().transpile().pattern

        if pauli_presimulate:
            pattern.remove_input_nodes()
            pattern = pattern.infer_pauli_measurements()
            pattern.perform_pauli_measurements()
        if min_space:
            pattern.minimize_space()

        return pattern

    def characterize(self, beautify: bool = True) -> pd.DataFrame:
        data: dict[str, str | int] = {}

        data["benchmark"] = self.name.value
        data["nqubits"] = self.nqubits

        circuit = self.to_circuit()
        data["n_gates"] = len(circuit.instruction)

        pattern = circuit.transpile().pattern
        pattern.remove_input_nodes()

        data["transp-max_space"] = pattern.max_space()
        data["transp-n_commands"] = len(pattern)

        pattern = pattern.infer_pauli_measurements()
        pattern.perform_pauli_measurements()
        data["pauli_ps-max_space"] = pattern.max_space()
        data["pauli_ps-n_commands"] = len(pattern)

        pattern.minimize_space()
        data["sp_min-max_space"] = pattern.max_space()
        data["sp_min-n_commands"] = len(pattern)

        df = pd.DataFrame([data])

        if beautify:
            return beautify_benchmark_df(df)
        return df


@dataclass
class BenchmarkResult:
    benchmark: Benchmark
    backend_name: str
    stats: pd.DataFrame

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> BenchmarkResult:
        benchmark_name = BenchmarkName[data["params"]["benchmark_name"].upper()]
        nqubits = int(data["params"]["nqubits"])
        benchmark = Benchmark(benchmark_name, nqubits)
        backend_name = data["extra_info"]["backend_name"]
        stats_df = pd.DataFrame([data["stats"]])

        return BenchmarkResult(benchmark, backend_name, stats_df)


def characterize_all_benchmarks(nqubits: int) -> pd.DataFrame:
    rows = []
    for bench in BenchmarkName:
        try:
            df = Benchmark(bench, nqubits).characterize(beautify=False)
        except ValueError:
            df = pd.DataFrame(
                [
                    {
                        "benchmark": bench.name,
                        "nqubits": nqubits,
                        "n_gates": np.nan,
                        "transp-max_space": np.nan,
                        "transp-n_commands": np.nan,
                        "pauli_ps-max_space": np.nan,
                        "pauli_ps-n_commands": np.nan,
                        "sp_min-max_space": np.nan,
                        "sp_min-n_commands": np.nan,
                    }
                ]
            )

        rows.append(df)

    return beautify_benchmark_df(pd.concat(rows, ignore_index=True, sort=False))


def characterize_benchmarks(benchmarks: Sequence[Benchmark]) -> pd.DataFrame:
    return beautify_benchmark_df(
        pd.concat([benchmark.characterize(beautify=False) for benchmark in benchmarks], ignore_index=True, sort=False)
    )


def beautify_benchmark_df(df: pd.DataFrame) -> pd.DataFrame:
    new_columns = []

    for col in df.columns:
        if col != "benchmark":
            df[col] = df[col].astype("Int64")
        if col in {"benchmark", "nqubits", "n_gates"}:
            new_columns.append(("Circuit", col))
        elif col.startswith("transp-"):
            new_columns.append(("After transpilation", col.replace("transp-", "")))
        elif col.startswith("pauli_ps-"):
            new_columns.append(("After Pauli presimulation", col.replace("pauli_ps-", "")))
        elif col.startswith("sp_min-"):
            new_columns.append(("After space minimization", col.replace("sp_min-", "")))
        else:
            new_columns.append(("other", col))

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
        df["Backend"] = r.backend_name
        rows.append(df)

    long_df = pd.concat(rows, ignore_index=True)

    # `swaplevel(axis=1)` places `Backend` above data.
    return long_df.pivot_table(index=["Benchmark", "# Qubits"], columns="Backend", aggfunc="first").swaplevel(axis=1)
