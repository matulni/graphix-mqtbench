"""Convert MQT Bench circuits to Graphix format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphix.instruction import InstructionKind
from graphix_qasm_parser import OpenQASMParser
from qiskit import QuantumCircuit, transpile
from qiskit.qasm3 import dumps

if TYPE_CHECKING:
    from graphix import Circuit


def graphix_native_gates_to_qiskit() -> list[str]:
    """Get the list of Graphix native gates in terms of Qiskit gate names.

    To see the native gates of Graphix, see :func:`graphix.instruction.InstructionKind`.
    Note that the RZZ gate is implemented using CRZ in Qiskit, so it is replaced accordingly.

    Returns
    -------
    list[str]
        A list of native gate names as recognized by Qiskit.
    """
    native_qiskit_gates = [instr.name.lower() for instr in InstructionKind]

    # RZZ is implemented using CRZ
    native_qiskit_gates.remove("rzz")
    native_qiskit_gates.append("crz")

    # m is implemented using measure
    native_qiskit_gates.remove("m")
    native_qiskit_gates.append("measure")

    # i is implemented using identity
    native_qiskit_gates.remove("i")
    native_qiskit_gates.append("id")

    # cnot is implemented using cx
    native_qiskit_gates.remove("cnot")
    native_qiskit_gates.append("cx")

    return native_qiskit_gates


def convert(qiskit_circuit: QuantumCircuit) -> Circuit:
    """Convert a Qiskit QuantumCircuit to a Graphix Circuit.

    Parameters
    ----------
    qiskit_circuit : QuantumCircuit
        The Qiskit circuit to convert.

    Returns
    -------
    Circuit
        The converted circuit in Graphix format.
    """
    parser = OpenQASMParser()
    transpiled_circuit = transpile(
        qiskit_circuit,
        basis_gates=list(graphix_native_gates_to_qiskit()),
        optimization_level=0,
    )

    # To ensure that the register names are compatible with the QASM parser,
    # we create a new QuantumCircuit and compose the transpiled circuit onto it
    # which effectively resets register names
    transferred_circuit = QuantumCircuit(
        transpiled_circuit.num_qubits,
        transpiled_circuit.num_clbits,
    )
    transferred_circuit.compose(transpiled_circuit, inplace=True)

    qasm_str = dumps(transferred_circuit)

    return parser.parse_str(qasm_str)
