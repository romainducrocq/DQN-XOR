#include "dqn/replay_memory.hpp"

void replayMemory::ReplayMemoryNaive::store_transition(
    std::vector<float>& obs, int64_t action, float rew, bool done, std::vector<float>& new_obs)
{
    this->replay_buffer.emplace_back(obs, action, rew, done, new_obs);

    while(this->replay_buffer.size() > this->buffer_size){
        this->replay_buffer.pop_front();
    }
}

void replayMemory::ReplayMemoryNaive::sample_transition(
        std::array<std::reference_wrapper<replayMemory::Transition>, CONF::BS> transitions)
{
    for(size_t i = 0; i < transitions.size(); i++){
        transitions[i] = this->replay_buffer.at(std::rand() % this->replay_buffer.size());
    }
}
