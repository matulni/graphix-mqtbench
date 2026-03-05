from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
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


class Test:
    # _SHORT_BENCHMARKS = (Benchmark(bench, 5) for bench in BenchmarkName)

    @pytest.mark.benchmark(max_time=1, min_rounds=3, warmup=True)
    @pytest.mark.parametrize("mqt_benchmark", prepare_benchmarks(5))
    def test_simulator(self, benchmark: BenchmarkFixture, mqt_benchmark: Benchmark) -> None:
        if mqt_benchmark is not None:
            runner = BenchmarkRunner(
                benchmark=mqt_benchmark,
                benchmark_fixture=benchmark,
                optim=OptimizationPass.M,
                backend=StatevectorBackend(),
                backend_name="statevector",
            )
            runner.run()
