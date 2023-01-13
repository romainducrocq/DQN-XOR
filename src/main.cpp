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
                network::DeepQNetwork deepQNetwork(device);
                network::DuelingDeepQNetwork duelingDeepQNetwork(device);
                // net.example();

                /*
                std::array<float, 27> obs = { 0.f };
                for(size_t i = 0; i < obs.size(); i++){
                    obs[i] = static_cast<float>(i+1);
                }
                std::vector<int64_t> input_dim = { 3, 3, 3 };

                torch::Tensor tensor = torch::from_blob(obs.data(), input_dim,
                    torch::TensorOptions().dtype(torch::kFloat32)).clone(); // .to(this->device);
                std::cout << tensor << std::endl;
                 */
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
