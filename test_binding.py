import sys
import numpy as np
import pytest
from scipy.signal import argrelmax
from pathlib import Path

from python_binding_nb.build.nb_ext import argrelmax as argrelmax_nb
from python_binding_pb.pb_ext import argrelmax as argrelmax_pb
from python_binding_pyo3 import argrelmax as argrelmax_pyo3


@pytest.mark.parametrize("num_exec", range(100))
@pytest.mark.parametrize("fn", [argrelmax_pb, argrelmax_nb, argrelmax_pyo3])
def test_pb(num_exec, fn):
    length = np.random.randint(10, 10000)
    order = np.random.randint(1, length)
    
    input = np.random.randn(
        int(length),
    ).astype(np.float32)
    gt = argrelmax(input, order=order)[0]
    arr = fn(input, order)

    assert np.array_equal(gt, arr)