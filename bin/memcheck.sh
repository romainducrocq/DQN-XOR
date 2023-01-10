#!/bin/bash

source utils.sh --source-specific get_args

args=$(get_args " ${@}")

if [ "$1" = "./main.sh" ] ; then
    args=" -m main"$(echo ${args} | cut -d'h' -f2-)
elif [ "$1" = "./test.sh" ] ; then
    args=" -m test"$(echo ${args} | cut -d'h' -f2-)
else
    exit
fi

./build.sh -D
./compile.sh

cd apps/
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=$(dirname $(dirname $(pwd)))/log/valgrind/out.txt \
         ./exec${args}

cat ../../log/valgrind/out.txt
