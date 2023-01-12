#ifndef _DQN_NETWORK_HPP
#define _DQN_NETWORK_HPP

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "conf.hpp"

namespace Network
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

    class Net : public torch::nn::Module
    {
        protected:
            const std::vector<int64_t>& input_dim = CONF::INPUTS;
            int64_t output_dim = CONF::OUTPUTS;
            double lr = CONF::LR;

            int64_t fc_output_dim = -1;

            torch::nn::Sequential net = nullptr;
            std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;
            // torch::Tensor(*loss)(const torch::Tensor& self, const torch::Tensor& target) = nullptr;

            void(*ctor_net)(const std::vector<int64_t>& input_dim,
                std::vector<int64_t>& hidden_dims, torch::nn::Sequential& net) = CONF::CTOR_NET;

            const torch::Device& device;

        public:
            Net(const torch::Device& device);

            // void example();

            /* TODO
            save
            load
            */

        protected:
            virtual torch::Tensor forward(torch::Tensor x) = 0;
            virtual int64_t action(std::vector<float>& obs) = 0;
    };

    class DeepQNet : public Network::Net
    {
        private:
            torch::nn::Linear fc_out { nullptr };

        public:
            DeepQNet(const torch::Device& device);

            torch::Tensor forward(torch::Tensor x) override;
            int64_t action(std::vector<float>& obs) override;
    };

    class DuelingDeepQNet : public Network::Net
    {
        private:
            torch::nn::Linear fc_val { nullptr };
            torch::nn::Linear fc_adv { nullptr };

        private:
            torch::Tensor value(torch::Tensor x);
            torch::Tensor advantages(torch::Tensor x);

        public:
            DuelingDeepQNet(const torch::Device& device);

            torch::Tensor forward(torch::Tensor x) override;
            int64_t action(std::vector<float>& obs) override;
    };
}

#endif