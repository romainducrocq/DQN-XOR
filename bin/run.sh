#!/bin/bash

source utils.sh --source-specific get_args

args=$(get_args " ${@}")

cd apps/
./exec${args}
