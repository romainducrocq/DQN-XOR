#!/bin/bash

./build.sh -R
./compile.sh
./run.sh -m test
./run.sh -m main
