#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

auto process(std::vector<bool> &results, py::array_t<float> &py_arr, const int shift,
             const int len) {
    for (auto i = 0; i < len; i++) {
        if (!results[i]) {
            continue;
        }
        const auto plus = *py_arr.data(std::min(i + shift, len - 1));
        const auto minus = *py_arr.data(std::max(i - shift, 0));
        const auto data = *py_arr.data(i);
        results[i] = results[i] && (data > plus) && (data > minus);
    }
}

auto argrelmax(py::array_t<float> &py_arr, const int order) {
    const auto &buff_info = py_arr.request();
    const auto &shape = buff_info.shape;
    const auto len = shape[0];

    auto results = std::vector<bool>(len, true);

    for (auto shift = 1; shift <= order; shift++) {
        process(results, py_arr, shift, len);
    }

    auto nonzero_array = std::vector<uint32_t>();
    nonzero_array.reserve(10000);
    for (auto i = 0; i < len; i++) {
        if (results[i]) {
            nonzero_array.emplace_back(static_cast<uint32_t>(i));
        }
    }
    return nonzero_array;
}

PYBIND11_MODULE(pb_ext, m) { m.def("argrelmax", &argrelmax, py::return_value_policy::move); }