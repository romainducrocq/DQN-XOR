#include "dqn/network.hpp"

network::Network::Network(const torch::Device& device)
    : device(device)
{
    std::vector<int64_t> fc_hid_dims;
    this->ctor_net(this->input_dim, fc_hid_dims, this->net);
    this->fc_output_dim = fc_hid_dims[fc_hid_dims.size()-1];

    this->register_module("net", this->net);
}

/*
void Network::Net::example()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}
*/

/* TODO
 * save
 * load
 * */

network::DeepQNetwork::DeepQNetwork(const torch::Device& device)
    : network::Network(device)
{
    this->fc_out = torch::nn::Linear(this->fc_output_dim, this->output_dim);

    this->register_module("fc_out", this->fc_out);

    this->optimizer = std::make_shared<torch::optim::Adam>(this->parameters(), this->lr);
    // Network::Optimizer::adam(this->optimizer, this->parameters(), this->lr);
    // this->loss = Network::Loss::smooth_l1_loss;

    this->to(this->device);
}

torch::Tensor network::DeepQNetwork::forward(torch::Tensor x)
{
    x = this->net->forward(x);
    x = this->fc_out->forward(x);
    return x;
}

int64_t network::DeepQNetwork::action(std::vector<float>& obs)
{
    auto obs_t = torch::from_blob(obs.data(), this->input_dim,
            torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);

    auto q_values = this->forward(obs_t.unsqueeze(0));
    auto max_q_index = torch::argmax(q_values, 1)[0];
    auto action = max_q_index.detach().item<int64_t>();

    return action;
}

network::DuelingDeepQNetwork::DuelingDeepQNetwork(const torch::Device& device)
    : network::Network(device)
{
    this->fc_val = torch::nn::Linear(this->fc_output_dim, 1);
    this->fc_adv = torch::nn::Linear(this->fc_output_dim, this->output_dim);

    this->register_module("fc_val", this->fc_val);
    this->register_module("fc_adv", this->fc_adv);

    this->optimizer = std::make_shared<torch::optim::Adam>(this->parameters(), this->lr);
    // Network::Optimizer::adam(this->optimizer, this->net->parameters(), this->lr);
    // this->loss = Network::Loss::smooth_l1_loss;

    this->to(this->device);
}

torch::Tensor network::DuelingDeepQNetwork::forward(torch::Tensor x)
{
    x = this->net->forward(x);
    auto val = this->fc_val->forward(x);
    auto adv = this->fc_adv->forward(x);
    x = torch::add(val, (adv - adv.mean(1, true)));
    return x;
}

torch::Tensor network::DuelingDeepQNetwork::value(torch::Tensor x)
{
    x = this->net->forward(x);
    x = this->fc_val->forward(x);
    return x;
}

torch::Tensor network::DuelingDeepQNetwork::advantages(torch::Tensor x)
{
    x = this->net->forward(x);
    x = this->fc_adv->forward(x);
    return x;
}

int64_t network::DuelingDeepQNetwork::action(std::vector<float>& obs)
{
    auto obs_t = torch::from_blob(obs.data(), this->input_dim,
            torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);

    auto adv_q_values = this->advantages(obs_t.unsqueeze(0));
    auto max_adv_q_index = torch::argmax(adv_q_values, 1)[0];
    auto action = max_adv_q_index.detach().item<int64_t>();

    return action;
}
