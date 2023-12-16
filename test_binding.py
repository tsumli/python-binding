import sys
import numpy as np
import pytest
from scipy.signal import argrelmax
from pathlib import Path

from python_binding_cpp.build.cpp_ext import argrelmax as cpp_argrelmax
from python_binding_cpp.build.cpp_ext import mp_argrelmax as cpp_mp_argrelmax
from python_binding_rust import argrelmax as rs_argrelmax
from python_binding_rust import mp_argrelmax as rs_mp_argrelmax



@pytest.mark.parametrize("execution_number", range(100))
def test_cpp(execution_number):
    length = np.random.randint(10, 10000)
    order = np.random.randint(1, length)
    
    input = np.random.randn(
        int(length),
    ).astype(np.float32)
    gt = argrelmax(input, order=order)[0]
    arr = cpp_argrelmax(input, order)

    assert np.array_equal(gt, arr)


@pytest.mark.parametrize("execution_number", range(100))
def test_rs(execution_number):
    length = np.random.randint(10, 10000)
    order = np.random.randint(1, length)
    
    input = np.random.randn(
        int(length),
    ).astype(np.float32)
    gt = argrelmax(input, order=order)[0]
    arr = rs_argrelmax(input, order)

    assert np.array_equal(gt, arr)
