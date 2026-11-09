// Python glue for NaiveSearch. Build: see setup.py.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "naive_search.h"

namespace py = pybind11;

// Any array-like is accepted and converted to contiguous float32 / uint8 if needed.
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using TagArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(naive_search, m) {
    py::class_<NaiveSearch>(m, "NaiveSearch")
        .def(py::init([](FloatArray vectors, TagArray tags) {
                 if (vectors.ndim() != 2 || tags.ndim() != 2 || vectors.shape(0) != tags.shape(0))
                     throw std::invalid_argument("expected vectors (N, D) and tags (N, C)");
                 return NaiveSearch(vectors.data(), tags.data(), (int)vectors.shape(0),
                                    (int)vectors.shape(1), (int)tags.shape(1));
             }),
             py::arg("vectors"), py::arg("tags"))
        .def("search",
             [](const NaiveSearch &self, FloatArray query, TagArray query_tags, int k) {
                 if (query.size() != self.dim() || query_tags.size() != self.num_tags())
                     throw std::invalid_argument("query must have D values, query_tags C values");
                 std::vector<int> ids(k > 0 ? k : 0);
                 std::vector<float> scores(ids.size());
                 int found = self.search(query.data(), query_tags.data(), k, ids.data(), scores.data());
                 ids.resize(found);
                 scores.resize(found);
                 return py::make_tuple(ids, scores);
             },
             py::arg("query"), py::arg("query_tags"), py::arg("k"));
}
