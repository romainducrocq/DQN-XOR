#include "dqn/agent.hpp"

agent::Agent::Agent(const torch::Device& device)
    : device(device)
{
    this->input_dim_bs.push_back(static_cast<int64_t>(this->batch_size));
    for(const auto& i : this->input_dim){
        this->input_dim_bs.push_back(i);
    }
}

void agent::Agent::sample_transitions_t()
{
    for(size_t t = 0; t < this->batch_size; t++){
        *this->transition = this->replay_memory.sample_transition();

        for(size_t o = 0; o < this->inputs; o++){
            this->obses_[this->inputs * t + o] = this->transition->obs[o];
            this->new_obses_[this->inputs * t + o] = this->transition->new_obs[o];
        }

        this->actions_[t] = this->transition->action;
        this->rews_[t] = this->transition->rew;
        this->dones_[t] = this->transition->done;
    }

    this->obses_t = torch::from_blob(this->obses_.data(), this->input_dim_bs,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);

    this->new_obses_t = torch::from_blob(this->new_obses_.data(), this->input_dim_bs,
        torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);

    this->actions_t = torch::from_blob(this->actions_.data(), { this->input_dim_bs[0] },
        torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);

    this->rews_t = torch::from_blob(this->rews_.data(), { this->input_dim_bs[0] },
        torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);

    this->dones_t = torch::from_blob(this->dones_.data(), { this->input_dim_bs[0] },
        torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);
}
