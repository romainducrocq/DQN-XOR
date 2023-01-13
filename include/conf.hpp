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

#include <torch/torch.h>

template<typename T>
struct DefaultConf{

    enum Mode{
        MAIN, TEST
    };

    static Mode MODE;

    const static int64_t INPUTS;
    const static int64_t OUTPUTS;

    const static double LR;

    const static size_t BS;
    const static size_t MAX_MEM;

    const static std::vector<int64_t> INPUT_DIM;

    static void CTOR_NET(const std::vector<int64_t>& input_dim,
        std::vector<int64_t>& hidden_dims, torch::nn::Sequential& net);

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
template<typename T>
const int64_t DefaultConf<T>::INPUTS = 3;
template<typename T>
const int64_t DefaultConf<T>::OUTPUTS = 2;
template<typename T>
const double DefaultConf<T>::LR = 0.01;

template<typename T>
const size_t DefaultConf<T>::BS = 32;
template<typename T>
const size_t DefaultConf<T>::MAX_MEM = 1000000;

template<typename T>
const std::vector<int64_t> DefaultConf<T>::INPUT_DIM = { 3 };

template<typename T>
void DefaultConf<T>::CTOR_NET(const std::vector<int64_t>& input_dim,
     std::vector<int64_t>& fc_hid_dims, torch::nn::Sequential& net)
{
    for(const auto& l : { 16, 16 }){
        fc_hid_dims.push_back(l);
    }

    auto activation = torch::nn::ELU();

    net = torch::nn::Sequential(
            torch::nn::Linear(input_dim[0], fc_hid_dims[0]),
            activation,
            torch::nn::Linear(fc_hid_dims[0], fc_hid_dims[1]),
            activation
    );
}

using CONF = DefaultConf<int>;

#endif
