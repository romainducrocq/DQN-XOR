#!/bin/bash

CONF_D=""

funcs=(
"assert"
"set_build_type"
"get_args"
"get_tr_lead_args"
"get_default_args"
"get_conf_param"
)

function assert () {
    if [ ${?} -ne 0 ]; then exit; fi;
}

function set_build_type () {
    BUILD_T="$4"
    if [ "$1" = "-R" ] || [ "$1" = "--Release" ] ; then
        BUILD_T="$2"
    elif [ "$1" = "-D" ] || [ "$1" = "--Debug" ] ; then
        BUILD_T="$3"
    fi
    echo "${BUILD_T}"
}

function get_args () {
    args=""
    for ARG in $( echo "${@}" | sed 's/ / /g' ) ; do
        args="${args} ${ARG}"
    done
    echo "${args}"
}

function get_tr_lead_args () {
    args="$1"
    if [ "$2" = "-R" ] || [ "$2" = "--Release" ] ; then
        args=$(echo ${args} | cut -d'R' -f2-)
    elif [ "$2" = "-D" ] || [ "$2" = "--Debug" ] ; then
        args=$(echo ${args} | cut -d'D' -f2-)
    fi
    echo "${args}"
}

function get_default_args () {
    args=$(cat '../include/'"${CONF_D}"'conf.hpp' | grep "$1" -A 1 | sed -n 2p | sed -e 's/^[[:space:]]*//')
    echo "${args}"
}

function get_conf_param () {
    param=$(cat '../include/'"${CONF_D}"'conf.hpp' | grep $(echo ':'"$1"' ') | tail -1 | cut -d'=' -f 2 | cut -d';' -f 1 | sed -e 's/^[[:space:]]*//')
    echo "${param}"
}

# SOURCE FUNCS

for((i=0;i<${#funcs[@]};i++)) ; do
    declare -f "${funcs[$i]}" >/dev/null
    [ $? -eq 0 ]
done

if [ "$1" == "--source-specific" ] ; then
    for((i=0;i<${#funcs[@]};i++)) ; do
        for((j=2;j<=$#;j++)); do
            test "${funcs[$i]}" == "$(eval 'echo ${'$j'}')" && continue 2
            done
        unset ${funcs[$i]}
    done
fi
unset i j funcs
