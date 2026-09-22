"""Definition of the MQTBenchmark wrapper."""

from __future__ import annotations

import contextlib
from functools import cached_property
from typing import TYPE_CHECKING, TypedDict

from mqt.bench import get_benchmark_indep
from qiskit.converters import circuit_to_dag, dag_to_circuit  # type: ignore[attr-defined]

from graphix_mqtbench._benchmark_names import BenchmarkName
from graphix_mqtbench.converter import qiskit_to_graphix_circuit

if TYPE_CHECKING:
    from graphix.pattern import Pattern
    from graphix.transpiler import Circuit
    from qiskit import QuantumCircuit
    from typing_extensions import Unpack  # Added in Python 3.11


class MQTBenchmark:
    """A wrapper around MQT Bench quantum circuits for benchmarking purposes.

    Retrieves a named benchmark circuit for a given qubit count and exposes
    it as a raw Qiskit circuit, and a Graphix circuit and pattern.

    Parameters
    ----------
    name : BenchmarkName
        The benchmark algorithm to instantiate.
    nqubits : int
        Number of qubits for the benchmark circuit.
    **kwargs
        Additional keyword arguments forwarded to :func:`mqt.bench.get_benchmark_indep`.

    Raises
    ------
    BenchmarkError
        If the requested benchmark does not exist for the given qubit count.

    Notes
    -----
    The raw Qiskit circuit from ``mqt.bench`` may include a terminal measurement segment, which is omitted from circuit as Graphix does not yet support classical registers. The circuit is also transpiled via :func:`qiskit.transpiler` into the gate set supported by Graphix, which may increase the gate count. Benchmarks ``GHZ_DYNAMIC``, ``SEVEN_QUBIT_STEANE_CODE``, and ``SHORS_NINE_QUBIT_CODE`` are not supported
    because they contain feed-forward primitives that are still unrepresentable on Graphix circuits.

    Examples
    --------
    >>> bench = MQTBenchmark(BenchmarkName.QFT, nqubits=4)
    >>> bench.name
    <BenchmarkName.QFT: 'qft'>
    >>> bench.nqubits
    4
    """

    _raw_circuit: QuantumCircuit
    _qiskit_circuit: QuantumCircuit

    def __init__(self, name: BenchmarkName, nqubits: int, **kwargs: Unpack[MQTBenchmarkKwargs]) -> None:
        self._name = name
        self._nqubits = nqubits
        self._kwargs = kwargs
        if name in {
            BenchmarkName.GHZ_DYNAMIC,
            BenchmarkName.SEVEN_QUBIT_STEANE_CODE,
            BenchmarkName.SHORS_NINE_QUBIT_CODE,
        }:
            # These benchmarks contain feed-forward that we can't represent on Graphix circuits yet.
            # Directly transpiling from qiskit to patterns is not supported yet
            raise BenchmarkError(f"{name.value} benchmark is not available on graphix-mqtbench yet.")
        try:
            self._raw_circuit = get_benchmark_indep(benchmark=name.value, circuit_size=nqubits, **kwargs)
            # Clear the layout before removing measurements to avoid qiskit warnings
            self._qiskit_circuit = dag_to_circuit(circuit_to_dag(self._raw_circuit))  # type: ignore[no-untyped-call]
            self._qiskit_circuit.remove_final_measurements()
        except Exception as e:
            # If a benchmark does not exist for a given number of qubits, it may raise several
            # different types of exceptions so we use a catch-all `Exception`.
            raise BenchmarkError(f"{name.value} benchmark does not exist for {nqubits} qubits.") from e

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the benchmark.

        Returns
        -------
        str
        """
        kwargs_str = (", " + ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())) if self._kwargs else ""
        return f"MQTBenchmark(name={self._name}, nqubits={self._nqubits}{kwargs_str})"

    @property
    def name(self) -> BenchmarkName:
        """The benchmark algorithm name.

        Returns
        -------
        BenchmarkName
            Enum member identifying the benchmark.
        """
        return self._name

    @property
    def nqubits(self) -> int:
        """The number of qubits in the benchmark circuit.

        Returns
        -------
        int
            Qubit count passed at construction.
        """
        return self._nqubits

    @property
    def raw_circuit(self) -> QuantumCircuit:
        """Target-independent raw Qiskit circuit for this benchmark.

        Prefer ``circuit`` for downstream Graphix work; this property
        exposes the underlying backend representation for interoperability.

        Returns
        -------
        QuantumCircuit
            Circuit as returned by :func:`mqt.bench.get_benchmark_indep`.
        """
        return self._raw_circuit

    @property
    def qiskit_circuit(self) -> QuantumCircuit:
        """The Qiskit circuit prepared for benchmarking.

        Some benchmarks are given with measurements at the end.
        We remove them to obtain a statevector.
        """
        return self._qiskit_circuit

    @cached_property
    def qiskit_circuit(self) -> QuantumCircuit:
        """The Qiskit circuit prepared for benchmarking.

        Some benchmarks are given with measurements at the end.
        We remove them to obtain a statevector.
        """
        qc_qiskit = self.raw_circuit

        # Clear the layout before removing measurements to avoid qiskit warnings
        qc_clean: QuantumCircuit = dag_to_circuit(circuit_to_dag(qc_qiskit))  # type: ignore[no-untyped-call]

        qc_clean.remove_final_measurements()
        return qc_clean

    @cached_property
    def circuit(self) -> Circuit:
        """The benchmark circuit converted to a Graphix :class:`graphix.transpiler.Circuit`.

        Computed once and cached on first access.

        Returns
        -------
        Circuit
            Graphix circuit for the benchmark.

        """
        return qiskit_to_graphix_circuit(self.qiskit_circuit)

    @cached_property
    def pattern(self) -> Pattern:
        """The MBQC pattern derived from the benchmark circuit.

        Transpiles ``circuit`` to a class:`graphix.pattern.Pattern` with default transpiler
        and caches the result.

        Returns
        -------
        Pattern
            MBQC pattern corresponding to ``circuit``.
        """
        return self.circuit.transpile().pattern


def generate_benchmarks(nqubits: int, **kwargs: Unpack[MQTBenchmarkKwargs]) -> tuple[MQTBenchmark, ...]:
    """Generate a sequence of all MQT benchmarks with a given number of qubits.

    If a given benchmark does not exist for ``nqubits``, it is skipped.

    Parameters
    ----------
    nqubits : int
        Number of qubits
    **kwargs
        Additional keyword arguments forwarded to :func:`mqt.bench.get_benchmark_indep`.

    Returns
    -------
    tuple[MQTBenchmark, ...]
    """
    benchmarks: list[MQTBenchmark] = []
    for bench in BenchmarkName:
        with contextlib.suppress(BenchmarkError):
            benchmarks.append(MQTBenchmark(bench, nqubits, **kwargs))
    return tuple(benchmarks)


class BenchmarkError(Exception):
    """Exception subclass to handle benchmark errors."""


class MQTBenchmarkKwargs(TypedDict, total=False):
    """Keyword arguments for initializing an ``MQTBenchmark``."""

    opt_level: int
    generate_mirror_circuit: bool
    random_parameters: bool
