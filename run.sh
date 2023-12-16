#!/bin/bash
docker-compose -f docker/docker-compose.yaml run --service-ports $1
