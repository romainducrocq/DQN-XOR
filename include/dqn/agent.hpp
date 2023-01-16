#ifndef _DQN_AGENT_HPP
#define _DQN_AGENT_HPP

#include <functional>

#include <array>
#include <vector>

#include <torch/torch.h>

#include "dqn/network.hpp"
#include "dqn/replay_memory.hpp"

#include "conf.hpp"

namespace agent
{
    class Agent
    {
        protected:
            size_t inputs = CONF::INPUTS;
            size_t batch_size = CONF::BS;

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

            replayMemory::ReplayMemoryNaive replay_memory;

            const std::vector<int64_t>& input_dim = CONF::INPUT_DIM;
            std::vector<int64_t> input_dim_bs = {};

            const torch::Device& device;

        protected:
            /*TODO*/

        public:
            Agent(const torch::Device& device);

        protected:
            void sample_transitions_t();
            //void transitions_to_tensor(std::vector<std::reference_wrapper<replayMemory::Transition>>& transitions,
            //    torch::Tensor& obses_t, torch::Tensor& actions_t, torch::Tensor& rews_t, torch::Tensor& dones_t, torch::Tensor& new_obses_t);
    };
}

#endif