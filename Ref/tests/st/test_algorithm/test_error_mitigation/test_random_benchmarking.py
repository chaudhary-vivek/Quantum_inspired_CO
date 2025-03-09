 
"""Test randomized benchmarking."""
import numpy as np
import pytest

from mindquantum.algorithm.error_mitigation import (
    generate_double_qubits_rb_circ,
    generate_single_qubit_rb_circ,
)


@pytest.mark.parametrize("method", [generate_single_qubit_rb_circ, generate_double_qubits_rb_circ])
def test_random_benchmarking_circuit(method):
    """
    Description: Test test_random_benchmarking_circuit
    Expectation: success
    """
    length = np.random.randint(0, 100, size=100)
    for i in length:
        circ = method(i, np.random.randint(1 << 20))
        assert np.allclose(np.abs(circ.get_qs())[0], 1, atol=1e-6)
