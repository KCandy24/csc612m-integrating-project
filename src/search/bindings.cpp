// Python glue for NaiveSearch. Build: see setup.py.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "naive_search.h"

namespace py = pybind11;

// Any array-like is accepted and converted to contiguous float32 / uint32 if needed.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using MaskArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(naive_search, m) {
    py::class_<NaiveSearch>(m, "NaiveSearch")
        .def(py::init([](FloatArray vectors, MaskArray masks) {
                 if (vectors.ndim() != 2 || masks.ndim() != 2 || vectors.shape(0) != masks.shape(0))
                     throw std::invalid_argument("expected vectors (N, D) and masks (N, W)");
                 return NaiveSearch(vectors.data(), masks.data(), (int)vectors.shape(0),
                                    (int)vectors.shape(1), (int)masks.shape(1));
             }),
             py::arg("vectors"), py::arg("masks"))
        .def("search",
             [](const NaiveSearch &self, FloatArray query, MaskArray query_mask, int k) {
                 if (query.size() != self.dim() || query_mask.size() != self.words())
                     throw std::invalid_argument("query must have D values, query_mask W words");
                 std::vector<int> ids(k > 0 ? k : 0);
                 std::vector<float> scores(ids.size());
                 int found = self.search(query.data(), query_mask.data(), k, ids.data(), scores.data());
                 ids.resize(found);
                 scores.resize(found);
                 return py::make_tuple(ids, scores);
             },
             py::arg("query"), py::arg("query_mask"), py::arg("k"));
}
