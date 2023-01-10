#!/bin/bash

source utils.sh --source-specific get_default_args get_conf_param

echo -n "" > README.md

while IFS="" read -r LINE || [ -n "$LINE" ]
do
    if [[ "${LINE}" == *"how_to_run"* ]] ; then
        echo "${LINE}" >> README.md
        echo '```' >> README.md
        echo '$ ./build.sh -R' >> README.md
        echo '$ ./compile.sh' >> README.md
        echo '$ ./run.sh -m test '"$(get_default_args 'TEST')" >> README.md
        echo '$ ./run.sh -m main '"$(get_default_args 'MAIN')" >> README.md
        echo '```' >> README.md
        echo '</span>' >> README.md
    elif [[ "${LINE}" == *"how_to_main"* ]] ; then
        echo "${LINE}" >> README.md
        echo '```' >> README.md
        echo '* Debug   : $ ./main.sh -D '"$(get_default_args 'MAIN')" >> README.md
        echo '* Release : $ ./main.sh -R '"$(get_default_args 'MAIN')" >> README.md
        echo '```' >> README.md
        echo '</span>' >> README.md
    elif [[ "${LINE}" == *"how_to_test"* ]] ; then
        echo "${LINE}" >> README.md
        echo '```' >> README.md
        echo '* Debug   : $ ./test.sh -D '"$(get_default_args 'TEST')" >> README.md
        echo '* Release : $ ./test.sh -R '"$(get_default_args 'TEST')" >> README.md
        echo '```' >> README.md
        echo '</span>' >> README.md
    fi

    if [[ "${LINE}" == *"span"* ]] ; then
        while IFS="" read -r LINE || [ -n "$LINE" ]
        do
            if [[ "${LINE}" == *"span"* ]] ; then
                break
            fi
        done
    else
        echo "${LINE}" >> README.md
    fi

done < ../README.md

mv README.md ..
