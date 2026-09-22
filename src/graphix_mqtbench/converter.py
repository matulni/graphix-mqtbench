"""Tools to convert MQT Bench circuits to Graphix format."""

from __future__ import annotations

from typing import TYPE_CHECKING

import qiskit
from graphix.instruction import InstructionKind
from graphix_qasm_parser import OpenQASMParser
from qiskit.qasm3 import dumps as qiskit_qasm3_dumps
from typing_extensions import assert_never  # intrduced in Python 3.11

if TYPE_CHECKING:
    from graphix import Circuit


def instruction_to_qiskit_gate(instr: InstructionKind) -> str:
    """Map a :class:`graphix.instruction.InstructionKind` to its Qiskit gate name string.

    Most gates map to the lowercase enum name. Exceptions are
    ``CNOT`` → ``"cx"``, ``I`` → ``"id"``, ``RZZ`` → ``"rrz"``,
    and ``M`` → ``"measure"``.

    Parameters
    ----------
    instr : InstructionKind
        The Graphix instruction to convert.

    Returns
    -------
    str
        Qiskit gate name string for use with the Qiskit circuit API.

    Raises
    ------
    AssertionError
        If ``instr`` is not a handled ``InstructionKind`` variant
        (via ``assert_never``).
    """
    match instr:
        case (
            InstructionKind.CCX
            | InstructionKind.SWAP
            | InstructionKind.CSWAP
            | InstructionKind.CY
            | InstructionKind.CZ
            | InstructionKind.H
            | InstructionKind.S
            | InstructionKind.SDG
            | InstructionKind.T
            | InstructionKind.TDG
            | InstructionKind.SX
            | InstructionKind.SXDG
            | InstructionKind.X
            | InstructionKind.Y
            | InstructionKind.Z
            | InstructionKind.P
            | InstructionKind.U
            | InstructionKind.RX
            | InstructionKind.RY
            | InstructionKind.RZ
            | InstructionKind.CP
            | InstructionKind.CU
            | InstructionKind.CRX
            | InstructionKind.CRY
            | InstructionKind.CRZ
            | InstructionKind.GPHASE
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
        case InstructionKind.J:
            raise ValueError("Qiskit does not have a native J gate.")
        case InstructionKind.CJ:
            raise ValueError("Qiskit does not have a native CJ gate.")
        case InstructionKind.CONDINSTR:
            raise ValueError("Conditional instructions are not part of the gate set.")
        case _:
            assert_never(instr)


# The qiskit transpiler does not support not standard gates (`rzz`). This could be explored further.
# ValueError: Providing non-standard gates (rrz) through the ``basis_gates`` argument is not allowed. Use the ``target`` parameter instead. You can build a target instance using ``Target.from_configuration()`` and provide custom gate definitions with the ``custom_name_mapping`` argument.


_GRAPHIX_NATIVE_GATES = [
    instruction_to_qiskit_gate(instr)
    for instr in InstructionKind
    if instr not in {InstructionKind.RZZ, InstructionKind.J, InstructionKind.CJ, InstructionKind.GPHASE, InstructionKind.CONDINSTR}
]


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
    # This function was developped by @ACE07-Sev for the unitaryDESIGN 2025
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
