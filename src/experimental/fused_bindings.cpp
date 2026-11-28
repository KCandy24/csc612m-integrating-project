// Python glue for FusedSearch
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>

#include "fused_search.h"

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using MaskArray = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

// the python module is called fused_search: `from fused_search import FusedSearch`
PYBIND11_MODULE(fused_search, m) {
    py::class_<FusedSearch>(m, "FusedSearch")
        // FusedSearch(vectors, masks): checks the shapes, then copies the data to the gpu
        .def(py::init([](FloatArray vectors, MaskArray masks) {
                 if (vectors.ndim() != 2 || masks.ndim() != 2 || vectors.shape(0) != masks.shape(0))
                     throw std::invalid_argument("expected vectors (N, D) and masks (N, W)");
                 return std::make_unique<FusedSearch>(vectors.data(), masks.data(), (int)vectors.shape(0),
                                                      (int)vectors.shape(1), (int)masks.shape(1));
             }),
             py::arg("vectors"), py::arg("masks"))
        // search(query, query_mask, k, fused=True, cosine=False, per_thread=False)
        // -> (ids, scores), best first
        .def("search",
             [](FusedSearch &self, FloatArray query, MaskArray query_mask, int k, bool fused, bool cosine,
                bool per_thread) {
                 if (query.size() != self.dim() || query_mask.size() != self.words())
                     throw std::invalid_argument("query must have D values, query_mask W words");
                 FusedSearch::Options options;
                 options.fused = fused;
                 options.cosine = cosine;
                 options.per_thread = per_thread;
                 // room for k results, then trim to how many were found
                 std::vector<int> ids(k > 0 ? k : 0);
                 std::vector<float> scores(ids.size());
                 int found = self.search(query.data(), query_mask.data(), k, options, ids.data(), scores.data());
                 ids.resize(found);
                 scores.resize(found);
                 return py::make_tuple(ids, scores);
             },
             py::arg("query"), py::arg("query_mask"), py::arg("k"), py::arg("fused") = true,
             py::arg("cosine") = false, py::arg("per_thread") = false)
        .def_property_readonly("last_kernel_ms", &FusedSearch::last_kernel_ms);
}
