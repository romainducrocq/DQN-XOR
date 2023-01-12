#include "dqn/network.hpp"

/* TODO to conf */
void ctor_net(int64_t input_dim, std::vector<int64_t>& hidden_dims, torch::nn::Sequential& net)
{
    // NETWORK
    hidden_dims.push_back(16);
    hidden_dims.push_back(16);

    auto activation = torch::nn::ELU();

    net = torch::nn::Sequential(
        torch::nn::Linear(input_dim, hidden_dims[0]),
        activation,
        torch::nn::Linear(hidden_dims[0], hidden_dims[1]),
        activation
    );
}
/**/

Network::Net::Net(const torch::Device& device)
    : device(device)
{
    std::vector<int64_t> h_dims;
    ctor_net(this->input_dim, h_dims, this->net);
    this->fc_out_dim = h_dims[h_dims.size() - 1];
    this->net = register_module("net", this->net);
}

/* TODO
 * save
 * load
 * */

Network::DQNet::DQNet(const torch::Device& device)
        : Network::Net(device)
{
    this->fc_out = torch::nn::Linear(this->fc_out_dim, this->output_dim);
    this->fc_out = register_module("fc_out", this->fc_out);

    Network::Optimizer::adam(this->optimizer, this->net->parameters(), this->lr);
    this->loss = Network::Loss::smooth_l1_loss;

    this->to(this->device);
}

torch::Tensor Network::DQNet::forward(torch::Tensor x)
{
    x = this->net->forward(x);
    x = this->fc_out->forward(x);
    return x;
}

size_t Network::DQNet::action(const std::vector<float>& obs)
{
    // https://github.com/pytorch/pytorch/blob/b3f0297a94977636fd90c0fe6fa9b971ff9f81e2/aten/src/ATen/native/quantized/cpu/conv_serialization.h#L116
    auto obs_t = torch::from_blob((float*)(obs.data()), static_cast<int64_t>(obs.size()),
                                  torch::TensorOptions().dtype(torch::kFloat32)).clone().to(this->device);
    auto q_values = this->forward(obs_t.unsqueeze(0));

    auto max_q_index = torch::argmax(q_values, 1)[0];
    auto action = max_q_index.detach().item<int64_t>();

    return action;
}

/*
Network::D3QNet::D3QNet()
    : Network::Net()
{
}

torch::Tensor Network::D3QNet::forward(torch::Tensor x)
{
    return x;
}

void Network::D3QNet::example()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}
*/
