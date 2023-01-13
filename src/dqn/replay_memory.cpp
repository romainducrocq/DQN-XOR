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
    std::vector<std::reference_wrapper<replayMemory::Transition>>& transitions)
{
    for(size_t i = 0; i < this->batch_size; i++){
        transitions.push_back(std::ref(this->replay_buffer.at(std::rand() % this->replay_buffer.size())));
    }
}
