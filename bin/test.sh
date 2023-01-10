#!/bin/bash

source utils.sh --source-specific set_build_type get_args get_tr_lead_args get_default_args

BUILD_T=$(set_build_type "$1" -R -D -R)

args=$(get_args " ${@}")

args=$(get_tr_lead_args "${args}" "$1")

if [ ${#args} -eq 0 ] ; then
    args=" "$(get_default_args "TEST")
fi

./build.sh ${BUILD_T}
./compile.sh
./run.sh -m test${args}
