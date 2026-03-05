from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend

from graphix_mqtbench import Benchmark, BenchmarkName, BenchmarkRunner, OptimizationPass

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


def prepare_benchmarks(nqubits: int) -> list[Benchmark | None]:
    tests: list[Benchmark | None] = []
    for bench in BenchmarkName:
        try:
            benchmark = Benchmark(bench, nqubits)
        except ValueError:
            benchmark = None
        tests.append(benchmark)
    return tests


class BenchTest:  # We use this name to allow for discovery. See .toml
    _SHORT_BENCHMARKS = (Benchmark(BenchmarkName.QFT, 3),)
    nqubits = 6

    @pytest.mark.benchmark(max_time=1, min_rounds=3, warmup=True)
    @pytest.mark.parametrize("mqt_benchmark", prepare_benchmarks(nqubits))
    def bench_statevector(self, benchmark: BenchmarkFixture, mqt_benchmark: Benchmark) -> None:
        if mqt_benchmark is not None:
            runner = BenchmarkRunner(
                benchmark=mqt_benchmark,
                benchmark_fixture=benchmark,
                optim=OptimizationPass.M,
                backend=StatevectorBackend(),
                backend_name="statevector",
            )
            runner.run()

    @pytest.mark.benchmark(max_time=1, min_rounds=3, warmup=True)
    @pytest.mark.parametrize("mqt_benchmark", prepare_benchmarks(nqubits))
    def bench_density_matrix(self, benchmark: BenchmarkFixture, mqt_benchmark: Benchmark) -> None:
        if mqt_benchmark is not None:
            runner = BenchmarkRunner(
                benchmark=mqt_benchmark,
                benchmark_fixture=benchmark,
                optim=OptimizationPass.M,
                backend=DensityMatrixBackend(),
                backend_name="density_matrix",
            )
            runner.run()
