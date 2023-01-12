#include <cstdlib>
#include <ctime>

#include <gtest/gtest.h>

#include "conf.hpp"

#include "utils/logger.hpp"
#include "utils/timer.hpp"

#include "dqn/network.hpp"

namespace App
{
    class Main
    {
        private:
            inline Main()
            {
                Timer timer;

                std::cout << "\n";
                std::cout << "-------------------------------MAIN--------------------------------" << "\n";
                std::cout << "\n";

                Logger::info("Hello world!");

                torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
                Network::DeepQNet deepQNet(device);
                Network::DuelingDeepQNet duelingDeepQNet(device);
                // net.example();
            };

        public:
            Main(const Main &other) = delete;

            Main operator=(const Main &other) = delete;

            static Main &MAIN()
            {
                static Main singleton;
                return singleton;
            }
    };
}

int main(int argc, char** argv)
{
    if(CONF::argParse(argc, argv)) {

        switch(CONF::MODE){

            case CONF::Mode::MAIN:{
                std::srand(time(nullptr));
                App::Main::MAIN();
                return 0;
            }

            case CONF::Mode::TEST:{
                std::srand(42);
                testing::InitGoogleTest(&argc, argv);
                return RUN_ALL_TESTS();
            }

            default:
                return 1;
        }

    }

    return 1;
}
