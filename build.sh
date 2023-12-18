#!/bin/bash
(cd ./python_binding_nb/ && . build.sh)
(cd ./python_binding_pb/ && . build.sh)
(cd ./python_binding_pyo3/ && . build.sh)
