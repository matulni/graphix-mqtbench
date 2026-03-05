from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd
from mqt.bench import get_benchmark_indep

from graphix_mqtbench._generated_benchmarks_enum import BenchmarkName
from graphix_mqtbench.converter import qiskit_to_graphix_circuit

if TYPE_CHECKING:
    from collections.abc import Mapping

    from graphix.pattern import Pattern
    from graphix.sim.base_backend import Backend
    from graphix.transpiler import Circuit
    from pytest_benchmark import BenchmarkFixture


class OptimizationPass(Enum):
    """Optimization passes on the transpiled pattern."""

    M = ("min_space",)
    """Minimize space."""

    P = ("pauli_presim",)
    "Pauli-presimulate."

    PM = ("pauli_presim", "min_space")
    """Pauli-presimulate, then minimize space."""

    def apply_optimization(self, pattern: Pattern) -> Pattern:
        pattern = pattern.copy()

        for step in self.value:
            match step:
                case "min_space":
                    pattern.minimize_space()
                case "pauli_presim":
                    pattern.remove_input_nodes()
                    pattern = pattern.infer_pauli_measurements()
                    pattern.perform_pauli_measurements()
        return pattern


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

    def to_pattern(self, optim: OptimizationPass | None = None) -> Pattern:
        pattern = self.to_circuit().transpile().pattern
        if optim is not None:
            pattern = optim.apply_optimization(pattern)
        return pattern

    def characterize(self, pretty: bool = True) -> pd.DataFrame:

        # To avoid circular imports
        from graphix_mqtbench import characterize_benchmark  # noqa: PLC0415

        return characterize_benchmark(self, pretty)


@dataclass
class BenchmarkRunner:
    benchmark: Benchmark
    benchmark_fixture: BenchmarkFixture
    optim: OptimizationPass | None
    backend: Backend
    backend_name: str

    def __post_init__(self) -> None:
        self.annotate_fixture()

    def annotate_fixture(self) -> None:
        self.benchmark_fixture.params = {"benchmark_name": self.benchmark.name, "nqubits": self.benchmark.nqubits}
        self.benchmark_fixture.extra_info = {
            "backend_name": self.backend_name,
            "optim": self.optim.name if self.optim is not None else None,
        }

    def run(self):
        pattern = self.benchmark.to_pattern(self.optim)

        def simulate():
            backend = self.backend.__class__()  # Initialize the backend for each run.
            return pattern.simulate_pattern(backend=backend)

        return self.benchmark_fixture(simulate)


@dataclass
class BenchmarkResult:
    benchmark: Benchmark
    extra_info: dict[str, str]
    stats: pd.DataFrame

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> BenchmarkResult:
        benchmark_name = BenchmarkName[data["params"]["benchmark_name"].upper()]
        nqubits = int(data["params"]["nqubits"])
        benchmark = Benchmark(benchmark_name, nqubits)
        extra_info = data["extra_info"]
        stats_df = pd.DataFrame([data["stats"]])

        return BenchmarkResult(benchmark, extra_info, stats_df)
