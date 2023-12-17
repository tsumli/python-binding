#!/bin/bash
docker build --network host -t python-binding:latest --file docker/Dockerfile .
