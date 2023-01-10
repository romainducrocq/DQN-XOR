#!/bin/bash

source utils.sh --source-specific set_build_type

BUILD_T=$(set_build_type "$1" Release Debug Debug)
PREFIX_P="/usr/lib/libtorch"

cmake -G "Unix Makefiles" -S ../build/ -B ../build/out/ -DCMAKE_BUILD_TYPE=${BUILD_T} -DCMAKE_PREFIX_PATH=${PREFIX_P}
