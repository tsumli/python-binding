# python-binding
Example usage of binding libraries.  
`scipy.signal.argrelmax` ([doc](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelmax.html)) is implemented in each library.

## Usage
```shell
sh docker/build.sh
sh run.sh main  # to measure times
sh run.sh test  # to test
sh run.sh jupyter  # to open jupyterlab
```

And you can check the comparison on `analysis.ipynb` for each `length` and `order`

## libraries
- pybind11
  -  https://github.com/pybind/pybind11
- nanobind
  -  https://github.com/wjakob/nanobind
- pyo3  
  - https://github.com/PyO3/pyo3
 
## TODO
- [ ] improve codes
- [ ] multithread
