#include "dqn/replay_memory.hpp"

void replayMemory::ReplayMemoryNaive::store_transition(
    std::vector<float>& obs, std::vector<float>& new_obs, int64_t action, float rew, bool done)
{
    this->replay_buffer.emplace_back(obs, new_obs, action, rew, done);

    while(this->replay_buffer.size() > this->buffer_size){
        this->replay_buffer.pop_front();
    }
}

const replayMemory::Transition& replayMemory::ReplayMemoryNaive::sample_transition()
{
    return this->replay_buffer.at(std::rand() % this->replay_buffer.size());
}
