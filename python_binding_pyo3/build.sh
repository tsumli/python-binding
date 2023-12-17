maturin build --release
pip install target/wheels/python_binding_pyo3-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl \
    --force-reinstall
