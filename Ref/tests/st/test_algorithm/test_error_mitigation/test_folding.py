 
"""Test folding circuit."""
import numpy as np

from mindquantum.algorithm.error_mitigation import fold_at_random
from mindquantum.utils import random_circuit


def test_folding_circuit():
    """
    Description: Test folding circuit
    Expectation: success
    """
    np.random.seed(42)
    circ = random_circuit(4, 20)
    locally_folding = fold_at_random(circ, 2.3)
    globally_folding = fold_at_random(circ, 2.3, 'globally')
    assert locally_folding[30].hermitian() == circ[12]
    assert locally_folding[30].hermitian() == globally_folding[25]
    matrix_1 = circ.matrix()
    matrix_2 = locally_folding.matrix()
    matrix_3 = globally_folding.matrix()
    assert np.allclose(matrix_1, matrix_2, atol=1e-4)
    assert np.allclose(matrix_1, matrix_3, atol=1e-4)
