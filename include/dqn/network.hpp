#ifndef _DQN_NETWORK_HPP
#define _DQN_NETWORK_HPP

#include <torch/torch.h>

#include <memory>

#include <vector>

namespace Network
{
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

    class Net : public torch::nn::Module
    {
        protected:
            int64_t input_dim = 3;
            int64_t output_dim = 2;
            int64_t fc_out_dim = 0;

            double lr = 0.01;

            torch::nn::Sequential net = nullptr;
            std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;
            torch::Tensor(*loss)(const torch::Tensor& self, const torch::Tensor& target) = nullptr;

            const torch::Device& device;

        public:
            Net(const torch::Device& device);

            /* TODO
            save
            load
            */

        protected:
            virtual torch::Tensor forward(torch::Tensor x) = 0;
            virtual size_t action(const std::vector<float>& obs) = 0;
    };

    class DQNet : public Network::Net
    {
        private:
            torch::nn::Linear fc_out { nullptr };

        private:
            torch::Tensor forward(torch::Tensor x) override;
            size_t action(const std::vector<float>& obs) override;

        public:
            DQNet(const torch::Device& device);
    };

    /*
    class D3QNet : public Network::Net
    {
        private:
            torch::Tensor forward(torch::Tensor x) override;

        public:
            D3QNet();

            void example();
    };
     */
}

#endif