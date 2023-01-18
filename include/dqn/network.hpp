#ifndef _DQN_NETWORK_HPP
#define _DQN_NETWORK_HPP

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "conf.hpp"

namespace network
{
    /*
    struct Optimizer
    {
        static inline void sgd(std::shared_ptr<torch::optim::Optimizer>& optimizer,
                               const std::vector<torch::Tensor>& params, const torch::optim::SGDOptions& defaults)
        {
            optimizer = std::make_shared<torch::optim::SGD>(params, defaults);
        }

        static inline void rmsprop(std::shared_ptr<torch::optim::Optimizer>& optimizer,
                                   const std::vector<torch::Tensor>& params, const torch::optim::RMSpropOptions& defaults={})
        {
            optimizer = std::make_shared<torch::optim::RMSprop>(params, defaults);
        }

        static inline void adam(std::shared_ptr<torch::optim::Optimizer>& optimizer,
                                const std::vector<torch::Tensor>& params, const torch::optim::AdamOptions& defaults={})
        {
            optimizer = std::make_shared<torch::optim::Adam>(params, defaults);
        }
    };

    struct Loss
    {
        static inline torch::Tensor l1_loss(const torch::Tensor& self, const torch::Tensor& target)
        {
            return torch::l1_loss(self, target, torch::Reduction::Mean);
        }

        static inline torch::Tensor mse_loss(const torch::Tensor& self, const torch::Tensor& target)
        {
            return torch::mse_loss(self, target, torch::Reduction::Mean);
        }

        static inline torch::Tensor smooth_l1_loss(const torch::Tensor& self, const torch::Tensor& target)
        {
            return torch::smooth_l1_loss(self, target, torch::Reduction::Mean, 1.0);
        }

        static inline torch::Tensor huber_loss(const torch::Tensor& self, const torch::Tensor& target)
        {
            return torch::huber_loss(self, target, torch::Reduction::Mean, 1.0);
        }
    };
     */

    class Network : public torch::nn::Module
    {
        protected:
            int64_t output_dim = CONF::OUTPUTS;
            double lr = CONF::LR;

            int64_t fc_output_dim = 0;

            torch::nn::Sequential net = nullptr;
            std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;
            // torch::Tensor(*loss)(const torch::Tensor& self, const torch::Tensor& target) = nullptr;

            const std::vector<int64_t>& input_dim = CONF::INPUT_DIM;

            void(*ctor_net)(const std::vector<int64_t>& input_dim,
                std::vector<int64_t>& hidden_dims, torch::nn::Sequential& net) = CONF::CTOR_NET;

            const torch::Device& device;

        public:
            Network(const torch::Device& device);
            virtual ~Network() = default;

            torch::optim::Optimizer& get_optimizer() const;

            // void example();

            /* TODO
            save
            load
            */

        public:
            virtual torch::Tensor forward(torch::Tensor x) = 0;
            virtual size_t action(std::vector<float>& obs) = 0;
    };

    class DeepQNetwork : public network::Network
    {
        private:
            torch::nn::Linear fc_out { nullptr };

        public:
            DeepQNetwork(const torch::Device& device);

            torch::Tensor forward(torch::Tensor x) override;
            size_t action(std::vector<float>& obs) override;
    };

    class DuelingDeepQNetwork : public network::Network
    {
        private:
            torch::nn::Linear fc_val { nullptr };
            torch::nn::Linear fc_adv { nullptr };

        private:
            torch::Tensor value(torch::Tensor x);
            torch::Tensor advantages(torch::Tensor x);

        public:
            DuelingDeepQNetwork(const torch::Device& device);

            torch::Tensor forward(torch::Tensor x) override;
            size_t action(std::vector<float>& obs) override;
    };
}

#endif