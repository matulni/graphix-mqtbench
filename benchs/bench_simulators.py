from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from graphix.sim.statevec import StatevectorBackend

from graphix_mqtbench import Benchmark, BenchmarkName

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


class Test:
    _SHORT_BENCHMARKS = (Benchmark(BenchmarkName.QFT, 14),)  # Benchmark(BenchmarkName.QFT, 6))

    @pytest.mark.benchmark(max_time=1, min_rounds=3, warmup=True)
    @pytest.mark.parametrize("benchmark_circuit", _SHORT_BENCHMARKS)
    def test_simulator(self, benchmark: BenchmarkFixture, benchmark_circuit: Benchmark) -> None:
        benchmark.params = {"benchmark_name": benchmark_circuit.name, "nqubits": benchmark_circuit.nqubits}
        benchmark.extra_info = {"backend_name": "statevector"}

        pattern = benchmark_circuit.to_pattern(pauli_presimulate=False, min_space=False)

        def simulate():
            backend = StatevectorBackend()
            return (pattern.simulate_pattern(backend=backend),)

        benchmark(simulate)
