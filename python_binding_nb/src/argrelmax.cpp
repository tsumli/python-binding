#include <omp.h>

#include <iostream>
#include <typeinfo>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/vector.h"

namespace nb = nanobind;

auto process(std::vector<bool> &results,
             nb::ndarray<const float, nb::shape<nb::any>, nb::c_contig> &py_arr, const int shift,
             const size_t len) {
    auto arr_view = py_arr.view();
    for (auto i = 0; i < static_cast<int>(len); i++) {
        if (!results[i]) {
            continue;
        }
        const auto plus = arr_view(std::min(i + shift, static_cast<int>(len - 1)));
        const auto minus = arr_view(std::max(i - shift, 0));
        const auto data = arr_view(i);
        results[i] = (data > plus) && (data > minus);
    }
}

auto argrelmax(nb::ndarray<const float, nb::shape<nb::any>, nb::c_contig> py_arr, const int order) {
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

NB_MODULE(nb_ext, m) { m.def("argrelmax", &argrelmax, nb::rv_policy::move); }
