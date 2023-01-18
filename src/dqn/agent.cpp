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
        *this->transition = this->replay_memory->sample_transition();

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
        torch::TensorOptions().dtype(torch::kInt64)).clone().to(this->device).unsqueeze(-1);

    this->rews_t = torch::from_blob(this->rews_.data(), { this->input_dim_bs[0] },
        torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device).unsqueeze(-1);

    this->dones_t = torch::from_blob(this->dones_.data(), { this->input_dim_bs[0] },
        torch::TensorOptions().dtype(torch::kBool)).toType(torch::kFloat32).clone().to(this->device).unsqueeze(-1);
}

float agent::Agent::epsilon() const
{
    return this->eps_dec_exp ?
       agent::Agent::Interp::exponential(this->step, 0, this->eps_dec, this->eps_start, this->eps_min) :
       agent::Agent::Interp::linear(this->step, 0, this->eps_dec, this->eps_start, this->eps_min);
}

size_t agent::Agent::choose_action(std::vector<float>& obs) const
{
    return (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) <= this->epsilon() ?
        std::rand() % this->outputs :
        this->online_network->action(obs);
}

void agent::Agent::update_target_network(bool force)
{
    if((! this->target_soft_update && this->step % this->target_update_freq == 0) || force){
        for(size_t i = 0; i < this->online_network->parameters().size(); i++){
            this->target_network->parameters()[i].data().copy_(this->online_network->parameters()[i].data());
        }
    }else{
        for(size_t i = 0; i < this->online_network->parameters().size(); i++){
            this->target_network->parameters()[i].data().copy_(
                (this->target_soft_update_tau * this->online_network->parameters()[i].data()) +
                ((1.f - this->target_soft_update_tau) * this->target_network->parameters()[i].data())
            );
        }
    }
}

/*
 * load_model
 * save_model
 * log
 * info_mean
 */

agent::SimpleAgent::SimpleAgent(const torch::Device &device)
    : agent::Agent(device)
{
}

void agent::SimpleAgent::learn()
{
    // Compute loss
    this->sample_transitions_t();

    torch::Tensor targets { nullptr };
    {
        torch::NoGradGuard no_grad;
        auto target_q_values = this->target_network->forward(this->new_obses_t);
        auto max_target_q_values = std::get<0>(target_q_values.max(1, true));

        targets = this->rews_t + ((1.f - this->dones_t) * (max_target_q_values * this->gamma));
    }

    auto online_q_values = this->online_network->forward(this->obses_t);
    auto action_q_values = torch::gather(online_q_values, 1, this->actions_t);

    auto loss = torch::smooth_l1_loss(action_q_values, targets, torch::Reduction::Mean, 1.0).to(this->device);

    // Gradient descent
    this->online_network->get_optimizer().zero_grad();
    loss.backward();
    this->online_network->get_optimizer().step();
}

agent::DoubleAgent::DoubleAgent(const torch::Device &device)
    : agent::Agent(device)
{
}

void agent::DoubleAgent::learn()
{
    // Compute loss
    this->sample_transitions_t();

    torch::Tensor targets { nullptr };
    {
        torch::NoGradGuard no_grad;
        auto target_online_q_values = this->online_network->forward(this->new_obses_t);
        auto target_online_best_q_indices = target_online_q_values.argmax(1, true);

        auto target_target_q_values = this->target_network->forward(this->new_obses_t);
        auto target_selected_q_values = torch::gather(target_target_q_values, 1, target_online_best_q_indices);

        targets = this->rews_t + ((1.f - this->dones_t) * (target_selected_q_values * this->gamma));
    }

    auto online_q_values = this->online_network->forward(this->obses_t);
    auto action_q_values = torch::gather(online_q_values, 1, this->actions_t);

    auto loss = torch::smooth_l1_loss(action_q_values, targets, torch::Reduction::Mean, 1.0).to(this->device);

    // Gradient descent
    this->online_network->get_optimizer().zero_grad();
    loss.backward();
    this->online_network->get_optimizer().step();
}

agent::DQNAgent::DQNAgent(const torch::Device &device)
     : agent::SimpleAgent(device)
{
    this->replay_memory = std::make_unique<replayMemory::ReplayMemoryNaive>();

    this->online_network = std::make_unique<network::DeepQNetwork>(device);
    this->target_network = std::make_unique<network::DeepQNetwork>(device);

    this->update_target_network(true);
}

agent::DoubleDQNAgent::DoubleDQNAgent(const torch::Device &device)
    : agent::DoubleAgent(device)
{
    this->replay_memory = std::make_unique<replayMemory::ReplayMemoryNaive>();

    this->online_network = std::make_unique<network::DeepQNetwork>(device);
    this->target_network = std::make_unique<network::DeepQNetwork>(device);

    this->update_target_network(true);
}

agent::DuelingDoubleDQNAgent::DuelingDoubleDQNAgent(const torch::Device &device)
    : agent::DoubleAgent(device)
{
    this->replay_memory = std::make_unique<replayMemory::ReplayMemoryNaive>();

    this->online_network = std::make_unique<network::DuelingDeepQNetwork>(device);
    this->target_network = std::make_unique<network::DuelingDeepQNetwork>(device);

    this->update_target_network(true);
}
