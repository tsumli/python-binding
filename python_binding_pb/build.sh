#!/bin/bash
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/argrelmax.cpp -std=c++2a -o pb_ext`python3-config --extension-suffix`
