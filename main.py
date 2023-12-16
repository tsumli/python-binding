import os
import pickle
import sys
import time
from typing import Dict, Tuple, Callable

import numpy as np
from scipy.signal import argrelmax
from tqdm import tqdm

from python_binding_cpp.build.cpp_ext import argrelmax as cpp_argrelmax
from python_binding_cpp.build.cpp_ext import mp_argrelmax as cpp_mp_argrelmax
from python_binding_rust import argrelmax as rs_argrelmax
from python_binding_rust import mp_argrelmax as rs_mp_argrelmax


def measure(fn: Callable, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start


def compare(input: np.ndarray, order: int) -> Dict[str, float]:
    ret = dict()
    ret["argrelmax"] = measure(argrelmax, input, order=order)
    ret["cpp_argrelmax"] = measure(cpp_argrelmax, input, order)
    # ret["cpp_mp_argrelmax"] = measure(cpp_mp_argrelmax, input, order)
    ret["rs_argrelmax"] = measure(rs_argrelmax, input, order)
    # ret["rs_mp_argrelmax"] = measure(rs_mp_argrelmax, input, order)
    return ret


if __name__ == "__main__":
    times: Dict[Tuple[int, int, int], Dict[str, float]] = dict()
    EXP = 1
    for length in tqdm(np.logspace(8, 8, 1)):
        for order in np.linspace(100, 100, 1):
            if order >= length:
                continue
            for exp in range(EXP):
                input = np.random.randn(
                    int(length),
                ).astype(np.float32)
                times[(length, order, exp)] = compare(input, int(order))
    with open("times_one.pkl", "wb") as f:
        pickle.dump(times, f)
