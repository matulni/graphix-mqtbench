from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import pytest
from graphix.sim.statevec import StatevectorBackend
from graphix.states import BasicStates
from mqt.bench import get_benchmark_indep
from qiskit.primitives import StatevectorSampler

from graphix_mqtbench import Benchmark, BenchmarkName, BenchmarkRunner, OptimizationPass

if TYPE_CHECKING:
    from graphix.transpiler import Circuit
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


def simulate_circuit(bench: Benchmark, shots: int, seed: int) -> tuple[dict[str, float], dict[str, float]]:
    qc_qiskit = get_benchmark_indep(bench.name.value, bench.nqubits)
    # We remove all measurements and measure again because not all qubits are measured in circuits of the benchmark suite.
    qc_qiskit.remove_final_measurements()
    qc_qiskit.measure_all()
    sampler = StatevectorSampler(seed=seed)
    creg_name = qc_qiskit.cregs[0].name
    result = sampler.run([qc_qiskit], shots=shots).result()[0].data
    counts_qiskit = getattr(result, creg_name).get_counts()
    prob_graphix = (
        bench.to_circuit().simulate_statevector(input_state=BasicStates.ZERO).statevec.to_prob_dict(encoding="MSB")
    )

    return counts_qiskit, prob_graphix


class CircuitTestCase(NamedTuple):
    benchmark: Benchmark
    circuit: Circuit


class TestBenchmark:
    SHOTS = 8096
    SEED = 24
    ERR = 0.02

    @pytest.mark.skip(reason="debug")
    @pytest.mark.parametrize("test_case", prepare_benchmarks(nqubits=2))
    def test_qiskit_simulation_2(self, test_case: Benchmark | None) -> None:
        if test_case is not None:
            counts_qiskit, prob_graphix = simulate_circuit(test_case, self.SHOTS, self.SEED)

            for key, value in counts_qiskit.items():
                assert math.isclose(prob_graphix[key], value / self.SHOTS, rel_tol=0, abs_tol=self.ERR)

    @pytest.mark.skip(reason="debug")
    @pytest.mark.parametrize("test_case", prepare_benchmarks(nqubits=3))
    def test_qiskit_simulation_3(self, test_case: Benchmark | None) -> None:
        if test_case is not None:
            counts_qiskit, prob_graphix = simulate_circuit(test_case, self.SHOTS, self.SEED)

            for key, value in counts_qiskit.items():
                assert math.isclose(prob_graphix[key], value / self.SHOTS, rel_tol=0, abs_tol=self.ERR)

    @pytest.mark.skip(reason="debug")
    @pytest.mark.parametrize("test_case", prepare_benchmarks(nqubits=4))
    def test_qiskit_simulation_4(self, test_case: Benchmark | None) -> None:
        if test_case is not None:
            counts_qiskit, prob_graphix = simulate_circuit(test_case, self.SHOTS, self.SEED)

            for key, value in counts_qiskit.items():
                assert math.isclose(prob_graphix[key], value / self.SHOTS, rel_tol=0, abs_tol=self.ERR)

    def test_to_pattern_m(self) -> None:
        b = Benchmark(BenchmarkName.QFT, 3)

        p = b.to_pattern(OptimizationPass.M)
        s = p.simulate_pattern()

        p_ref = b.to_circuit().transpile().pattern
        p_ref.minimize_space()
        s_ref = p_ref.simulate_pattern()

        assert s.isclose(s_ref)

    def test_to_pattern_p(self) -> None:
        b = Benchmark(BenchmarkName.GHZ, 3)

        p = b.to_pattern(OptimizationPass.P)
        s = p.simulate_pattern()

        p_ref = b.to_circuit().transpile().pattern
        p_ref.remove_input_nodes()
        p_ref = p_ref.infer_pauli_measurements()
        p_ref.perform_pauli_measurements()
        s_ref = p_ref.simulate_pattern()

        assert s.isclose(s_ref)

    def test_to_pattern_pm(self) -> None:
        b = Benchmark(BenchmarkName.BV, 2)

        p = b.to_pattern(OptimizationPass.PM)
        s = p.simulate_pattern()

        p_ref = b.to_circuit().transpile().pattern
        p_ref.remove_input_nodes()
        p_ref = p_ref.infer_pauli_measurements()
        p_ref.perform_pauli_measurements()
        p_ref.minimize_space()
        s_ref = p_ref.simulate_pattern()

        assert s.isclose(s_ref)

    def test_characterize(self) -> None:
        nqubits = 4
        bench = Benchmark(BenchmarkName.QFT, nqubits)
        pd = bench.characterize(
            optim_passses=[OptimizationPass.M, OptimizationPass.P, OptimizationPass.PM], pretty=False
        )

        circuit = bench.to_circuit()

        assert pd["nqubits"][0] == nqubits
        assert pd["n_gates"][0] == len(circuit.instruction)

        pattern = circuit.transpile().pattern
        assert pd["transp-max_space"][0] == pattern.max_space()
        assert pd["transp-n_commands"][0] == len(pattern)

        pattern_m = pattern.copy()
        pattern_m.minimize_space()
        assert pd["M-max_space"][0] == pattern_m.max_space()
        assert pd["M-n_commands"][0] == len(pattern_m)

        pattern.remove_input_nodes()
        pattern = pattern.infer_pauli_measurements()
        pattern.perform_pauli_measurements()
        assert pd["P-max_space"][0] == pattern.max_space()
        assert pd["P-n_commands"][0] == len(pattern)

        pattern.minimize_space()
        assert pd["PM-max_space"][0] == pattern.max_space()
        assert pd["PM-n_commands"][0] == len(pattern)


class TestBenchmarkRunner:
    @pytest.mark.benchmark(max_time=1, min_rounds=3, warmup=True)
    def test_benchmarkrunner(self, benchmark: BenchmarkFixture) -> None:

        mqt_benchmark = Benchmark(BenchmarkName.AE, nqubits=4)
        optim_pass = OptimizationPass.M
        runner = BenchmarkRunner(
            benchmark=mqt_benchmark,
            benchmark_fixture=benchmark,
            optim=optim_pass,
            backend=StatevectorBackend(),
            backend_name="test",
        )
        sv = runner.run()

        sv_ref = mqt_benchmark.to_pattern(optim_pass).simulate_pattern()

        assert sv.isclose(sv_ref)
