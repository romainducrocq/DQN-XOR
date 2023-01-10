#!/bin/bash

source utils.sh --source-specific get_args

args=$(get_args " ${@}")

if [ "$1" = "-V" ] || [ "$1" = "--Verbose" ] ; then
    if [ "$2" = "./main.sh" ] || [ "$2" = "./test.sh" ] ; then
        VERBOSE=true
    else
        exit
    fi
else
    if [ "$1" = "./main.sh" ] || [ "$1" = "./test.sh" ] ; then
        VERBOSE=false
    else
        exit
    fi
fi

if [ ${VERBOSE} = true ] ; then
    echo -n "" > ../log/logs/out.txt
    $(echo "${args}" | cut -d'V' -f2-) 2>&1 | tee -a ../log/logs/out.txt
else
    ${args} > ../log/logs/out.txt
    cat ../log/logs/out.txt | grep -e FATAL -e ERROR -e WARN -e INFO -e DEBUG -e TRACE
fi
