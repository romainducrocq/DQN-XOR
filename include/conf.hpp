#ifndef _ENV_CONF_HPP
#define _ENV_CONF_HPP

/*** DEF DEFAULT ARGS HERE */
/*
 * MAIN

 * TEST

*/

#include <unistd.h>

#include <iostream>
#include <cstddef>
#include <cstring>
#include <cstdlib>

template<typename T>
struct DefaultConf{

    enum Mode{
        MAIN, TEST
    };

    static Mode MODE;

    /*** DEC OPT PARAMS HERE */


    static inline bool argParse(int argc, char** argv)
    {
        // https://github.com/gnif/LookingGlass/blob/c0c63fd93bf999b6601a782fec8b56e9133388cc/client/main.c#L1391

        /*** DEF CMDS HERE */
        const char cmds[] = "h:m:";

        for(;;){
            switch(getopt(argc, argv, cmds)){

                /*** DEF HELP HERE */
                case '?': // help
                case 'h':
                default :
                    std::cerr << "usage: apps/exec [-h] [-m MOD] \n";
                    std::cerr << "\n";
                    std::cerr << "Template GTest                                                                       \n";
                    std::cerr << "\n";
                    std::cerr << "optional args:                                                                       \n";
                    std::cerr << "  -h      Print help and exit                                                        \n";
                    std::cerr << "  -m MOD  Set mode < main | test >                                                   \n";

                    return false;

                case -1:
                    break;

                case 'm': // mode < main | test >
                    if(std::strcmp(optarg, "main") == 0){
                        DefaultConf<T>::MODE = DefaultConf<T>::Mode::MAIN;
                    }else if(std::strcmp(optarg, "test") == 0){
                        DefaultConf<T>::MODE = DefaultConf<T>::Mode::TEST;
                    }
                continue;

                /*** DEF OPT CMDS HERE */


            }
            break;
        }

        return true;
    }
};

template<typename T>
typename DefaultConf<T>::Mode DefaultConf<T>::MODE = DefaultConf<T>::Mode::MAIN;

/*** DEF OPT PARAMS HERE */


using CONF = DefaultConf<int>;

#endif
