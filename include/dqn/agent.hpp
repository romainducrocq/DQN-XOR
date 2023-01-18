#ifndef _DQN_AGENT_HPP
#define _DQN_AGENT_HPP

#include <cstdlib>
#include <cmath>

#include <memory>

#include <array>
#include <vector>

#include <algorithm>

#include <torch/torch.h>

#include "dqn/network.hpp"
#include "dqn/replay_memory.hpp"

#include "conf.hpp"

namespace agent
{
    class Agent
    {
        private:
            struct Interp
            {
                static inline float linear(size_t x, size_t x1, size_t x2, float y1, float y2)
                {
                    return y1 + (y2 - y1) *
                        (static_cast<float>(std::min(std::max(x, x1), x2) - x1) / static_cast<float>(x2 - x1));
                }

                static inline float exponential(size_t x, size_t x1, size_t x2, float y1, float y2)
                {
                    return std::exp(std::log(y1) + (std::log(y2) - std::log(y1)) *
                        (static_cast<float>(std::min(std::max(x, x1), x2) - x1) / static_cast<float>(x2 - x1)));
                }
            };

        protected:
            size_t inputs = CONF::INPUTS;
            size_t outputs = CONF::OUTPUTS;

            size_t batch_size = CONF::BS;

            float gamma = CONF::GAMMA;

            float eps_start = CONF::EPS_START;
            float eps_min = CONF::EPS_MIN;
            size_t eps_dec = CONF::EPS_DEC;
            bool eps_dec_exp = CONF::EPS_DEC_EXP;

            size_t target_update_freq = CONF::TARGET_UPDATE_FREQ;
            bool target_soft_update = CONF::TARGET_SOFT_UPDATE;
            float target_soft_update_tau = CONF::TARGET_SOFT_UPDATE_TAU;

            const std::vector<int64_t>& input_dim = CONF::INPUT_DIM;
            std::vector<int64_t> input_dim_bs = {};

            size_t step = 0;

            /*transition to tensor*/
            replayMemory::Transition* transition = nullptr;

            std::array<float, CONF::INPUTS * CONF::BS> obses_ = { 0.f };
            std::array<float, CONF::INPUTS * CONF::BS> new_obses_ = { 0.f };
            std::array<int64_t, CONF::BS> actions_ = { 0 };
            std::array<float, CONF::BS> rews_ = { 0.f };
            std::array<bool, CONF::BS> dones_ = { false };

            torch::Tensor obses_t { nullptr };
            torch::Tensor new_obses_t { nullptr };
            torch::Tensor actions_t { nullptr };
            torch::Tensor rews_t { nullptr };
            torch::Tensor dones_t { nullptr };

            std::unique_ptr<replayMemory::ReplayMemory> replay_memory;
            std::unique_ptr<network::Network> online_network;
            std::unique_ptr<network::Network> target_network;

            const torch::Device& device;

        protected:
            /*TODO*/

            /**/

        public:
            Agent(const torch::Device& device);
            virtual ~Agent() = default;

        private:
            float epsilon() const;

        protected:
            void sample_transitions_t();
            size_t choose_action(std::vector<float>& obs) const;
            void update_target_network(bool force);

            virtual void learn() = 0;

            /*
            * load_model
            * save_model
            * log
            * info_mean
            */

            //void transitions_to_tensor(std::vector<std::reference_wrapper<replayMemory::Transition>>& transitions,
            //    torch::Tensor& obses_t, torch::Tensor& actions_t, torch::Tensor& rews_t, torch::Tensor& dones_t, torch::Tensor& new_obses_t);
    };

    class SimpleAgent : public agent::Agent
    {
        public:
            SimpleAgent(const torch::Device& device);

            void learn() override;
    };

    class DoubleAgent : public agent::Agent
    {
        public:
            DoubleAgent(const torch::Device& device);

            void learn() override;
    };

    class DQNAgent : public agent::SimpleAgent
    {
        public:
            DQNAgent(const torch::Device& device);
    };

    class DoubleDQNAgent : public agent::DoubleAgent
    {
        public:
            DoubleDQNAgent(const torch::Device& device);
    };

    class DuelingDoubleDQNAgent : public agent::DoubleAgent
    {
        public:
            DuelingDoubleDQNAgent(const torch::Device& device);
    };
}

#endif