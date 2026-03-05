"""Convert MQT Bench circuits to Graphix format."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import qiskit
from graphix.instruction import InstructionKind
from graphix_qasm_parser import OpenQASMParser
from qiskit.qasm3 import dumps as qiskit_qasm3_dumps

if TYPE_CHECKING:
    from graphix import Circuit


def instruction_to_qiskit_gate(instr: InstructionKind) -> str:
    match instr:
        case (
            InstructionKind.CCX
            | InstructionKind.SWAP
            | InstructionKind.CZ
            | InstructionKind.H
            | InstructionKind.S
            | InstructionKind.X
            | InstructionKind.Y
            | InstructionKind.Z
            | InstructionKind.RX
            | InstructionKind.RY
            | InstructionKind.RZ
        ):
            return instr.name.lower()
        case InstructionKind.RZZ:
            return "rrz"
        case InstructionKind.CNOT:
            return "cx"
        case InstructionKind.I:
            return "id"
        case InstructionKind.M:
            return "measure"
        case _:
            assert_never(instr)


# The qiskit transpiler does not support not standard gates (`rzz`). This could be explored further.
# ValueError: Providing non-standard gates (rrz) through the ``basis_gates`` argument is not allowed. Use the ``target`` parameter instead. You can build a target instance using ``Target.from_configuration()`` and provide custom gate definitions with the ``custom_name_mapping`` argument.


_GRAPHIX_NATIVE_GATES = [instruction_to_qiskit_gate(instr) for instr in InstructionKind if instr != InstructionKind.RZZ]


def qiskit_to_graphix_circuit(qiskit_circuit: qiskit.QuantumCircuit) -> Circuit:
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
    transpiled_circuit = qiskit.transpile(
        qiskit_circuit,
        basis_gates=_GRAPHIX_NATIVE_GATES,
        optimization_level=0,
    )
    # To ensure that the register names are compatible with the QASM parser,
    # we create a new QuantumCircuit and compose the transpiled circuit onto it
    # which effectively resets register names
    transferred_circuit = qiskit.QuantumCircuit(
        transpiled_circuit.num_qubits,
        transpiled_circuit.num_clbits,
    )
    transferred_circuit.compose(transpiled_circuit, inplace=True)

    qasm_str = qiskit_qasm3_dumps(transferred_circuit)

    parser = OpenQASMParser()
    return parser.parse_str(qasm_str)
