from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from graphix.sim.statevec import StatevectorBackend

from graphix_mqtbench import Benchmark, BenchmarkName

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


class Test:
    nqubits = 3
    _SHORT_BENCHMARKS = (Benchmark(BenchmarkName.AE, nqubits), Benchmark(BenchmarkName.DJ, nqubits))

    @pytest.mark.benchmark(group="statevector", max_time=1, min_rounds=5, warmup=True)
    @pytest.mark.parametrize("benchmark_circuit", _SHORT_BENCHMARKS)
    def test_simulator(self, benchmark: BenchmarkFixture, benchmark_circuit: Benchmark) -> None:
        benchmark.params = {"benchmark_name": benchmark_circuit.name, "nqubits": benchmark_circuit.nqubits}
        benchmark.extra_info = {"backend_name": "statevector"}

        pattern = benchmark_circuit.to_pattern(pauli_presimulate=True, min_space=True)

        def simulate():
            backend = StatevectorBackend()
            return (pattern.simulate_pattern(backend=backend),)

        benchmark(simulate)
