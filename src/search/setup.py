# Build in place:  python setup.py build_ext --inplace
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="naive_search",
    ext_modules=[Pybind11Extension("naive_search", ["bindings.cpp"])],
    cmdclass={"build_ext": build_ext},
)
