import os
import pickle
import sys
import time
from typing import Dict, Tuple, Callable

import numpy as np
from scipy.signal import argrelmax
from tqdm import tqdm

from python_binding_nb.build.cpp_ext import argrelmax as argrelmax_nb
from python_binding_pyo3 import argrelmax as argrelmax_pyo3


def measure(fn: Callable, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start


def compare(input: np.ndarray, order: int) -> Dict[str, float]:
    ret = dict()
    ret["orig"] = measure(argrelmax, input, order=order)
    ret["nb"] = measure(argrelmax_nb, input, order)
    ret["pyo3"] = measure(argrelmax_pyo3, input, order)
    return ret


if __name__ == "__main__":
    times: Dict[Tuple[int, int, int], Dict[str, float]] = dict()
    LENGTH_RANGE = np.logspace(5, 8, 10)
    ORDER_RANGE = np.linspace(10, 100, 10)
    NUM_EXP = 1
    for length in tqdm(LENGTH_RANGE):
        for order in ORDER_RANGE:
            if order >= length:
                continue
            for exp in range(NUM_EXP):
                input = np.random.randn(
                    int(length),
                ).astype(np.float32)
                times[(length, order, exp)] = compare(input, int(order))
    with open("times.pkl", "wb") as f:
        pickle.dump(times, f)
