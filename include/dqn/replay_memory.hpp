#ifndef _DQN_REPLAY_MEMORY_HPP
#define _DQN_REPLAY_MEMORY_HPP

#include <cstdlib>
#include <functional>

#include <vector>
#include <deque>

#include "conf.hpp"

namespace replayMemory
{
    struct Transition
    {
        std::vector<float> obs = {};
        int64_t action = 0;
        float rew = 0.f;
        bool done = false;
        std::vector<float> new_obs = {};

        Transition() = default;
        Transition(std::vector<float> obs, int64_t action, float rew, bool done, std::vector<float> new_obs)
            : obs(std::move(obs)), action(action), rew(rew), done(done), new_obs(std::move(new_obs)) {}
    };

    class ReplayMemory
    {
        protected:
            size_t batch_size = CONF::BS;
            size_t buffer_size = CONF::MAX_MEM;

        public:
            ReplayMemory() = default;

        protected:
            virtual void store_transition(
                std::vector<float>& obs, int64_t action, float rew, bool done, std::vector<float>& new_obs) = 0;
            virtual void sample_transitions(
                std::vector<std::reference_wrapper<replayMemory::Transition>>& transitions) = 0;
    };

    class ReplayMemoryNaive : public replayMemory::ReplayMemory
    {
        private:
            std::deque<replayMemory::Transition> replay_buffer;

        public:
            ReplayMemoryNaive() = default;

            void store_transition(
                std::vector<float>& obs, int64_t action, float rew, bool done, std::vector<float>& new_obs) override;
            void sample_transitions(
                std::vector<std::reference_wrapper<replayMemory::Transition>>& transitions) override;
    };
}

#endif