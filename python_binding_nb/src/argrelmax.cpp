#include <omp.h>

#include <iostream>
#include <typeinfo>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/vector.h"

namespace nb = nanobind;

auto process(std::vector<bool> &results, const nb::ndarray<const float, nb::shape<nb::any>> &py_arr,
             const int shift, const size_t len) {
    for (auto i = 0; i < static_cast<int>(len); i++) {
        if (!results[i]) {
            continue;
        }
        const auto plus = py_arr(std::min(i + shift, static_cast<int>(len - 1)));
        const auto minus = py_arr(std::max(i - shift, 0));
        const auto data = py_arr(i);
        results[i] = (data > plus) && (data > minus);
    }
}

auto argrelmax(const nb::ndarray<const float, nb::shape<nb::any>> &py_arr, const int order) {
    const auto len = py_arr.shape(0);
    auto results = std::vector<bool>(len, true);
    for (size_t shift = 1; shift <= static_cast<size_t>(order); shift++) {
        process(results, py_arr, shift, len);
    }

    auto nonzero_array = std::vector<uint32_t>();
    nonzero_array.reserve(10000);
    for (size_t i = 0; i < len; i++) {
        if (results[i]) {
            nonzero_array.emplace_back(static_cast<uint32_t>(i));
        }
    }

    // prepare returned value
    return nonzero_array;
}

NB_MODULE(cpp_ext, m) { m.def("argrelmax", &argrelmax, nb::rv_policy::move); }
