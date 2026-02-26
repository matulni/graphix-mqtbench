from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from graphix.pattern import Pattern
from graphix.transpiler import Circuit
from mqt.bench import get_benchmark_indep

from graphix_mqtbench._generated_benchmarks_enum import BenchmarkName
from graphix_mqtbench.converter import convert


@dataclass
class Benchmark:
    name: BenchmarkName
    nqubits: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        try:
            self.to_circuit()
        except Exception as e:
            raise ValueError(f"{self.name.value} benchmark does not exist for {self.nqubits} qubits.") from e

    def to_circuit(self) -> Circuit:
        return convert(
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
            return beautify_df(df)
        return df


def characterize_benchmarks(nqubits: int) -> pd.DataFrame:
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

    return beautify_df(pd.concat(rows, ignore_index=True, sort=False))


def beautify_df(df: pd.DataFrame) -> pd.DataFrame:
    new_columns = []

    for col in df.columns:
        if col != "benchmark":
            df[col] = df[col].astype("Int64")
        if col in ["benchmark", "nqubits", "n_gates"]:
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
    df = df.rename(
        columns={
            "n_commands": "# Commands",
            "max_space": "Max Space",
            "n_gates": "# Gates",
            "nqubits": "# Qubits",
            "benchmark": "Benchmark",
        },
        level=1,
    )

    return df
