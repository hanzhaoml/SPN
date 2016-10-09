//
// Created by Han Zhao on 7/13/15.
//

#ifndef SPN_EM_ONLINEPARAMLEARNING_H
#define SPN_EM_ONLINEPARAMLEARNING_H

#include "SPNetwork.h"

#include <vector>

namespace SPN {
    class OnlineParamLearning {
    public:
        OnlineParamLearning() = default;

        virtual ~OnlineParamLearning() = default;

        // Online Learning algorithm to be implemented by subclasses.
        virtual void fit(std::vector<std::vector<double>> &trains,
                         std::vector<std::vector<double>> &valids,
                         SPNetwork &spn, int num_iters, bool verbose = false) { };

        std::string algo_name() const {
            return algo_name_;
        }

    protected:
        std::string algo_name_ = "AbstractOnlineParamLearning";
    };

    // Projected Stochastic gradient descent algorithm for
    // learning the parameters of Sum-Product networks.
    class OnlineProjectedGD : public OnlineParamLearning {
    public:
        OnlineProjectedGD() : lrate_(1e-1), shrink_weight_(0.8),
                              proj_eps_(1e-2), stop_thred_(1e-3) {
            algo_name_ = "OnlineProjectedGD";
        }

        OnlineProjectedGD(double proj_eps, double stop_thred,
                          double lrate, double shrink_weight) : lrate_(lrate), shrink_weight_(shrink_weight),
                                                                proj_eps_(proj_eps), stop_thred_(stop_thred) {
            algo_name_ = "OnlineProjectedGD";
        }

        virtual ~OnlineProjectedGD() = default;

        void fit(std::vector<std::vector<double>> &trains,
                 std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, int num_iters=1, bool verbose = false) override;

    protected:
        double lrate_;
        double shrink_weight_;
        double proj_eps_;
        double stop_thred_;
    };

    // Exponentiated Stochastic gradient descent algorithm for
    // learning the parameters of Sum-Product Networks.
    class OnlineExpoGD : public OnlineParamLearning {
    public:
        OnlineExpoGD() : lrate_(1e-1), shrink_weight_(0.8), stop_thred_(1e-3) {
            algo_name_ = "OnlineExpoGD";
        }

        OnlineExpoGD(double stop_thred, double lrate, double shrink_weight) :
                stop_thred_(stop_thred), lrate_(lrate), shrink_weight_(shrink_weight) {
            algo_name_ = "OnlineExpoGD";
        }

        virtual ~OnlineExpoGD() = default;

        void fit(std::vector<std::vector<double>> &trains,
                 std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, int num_iters=1, bool verbose = false) override;

    protected:
        double lrate_;
        double shrink_weight_;
        double stop_thred_;
    };

    // Online stochastic version of the sequential monomial minimization algorithm.
    class OnlineSMA : public OnlineParamLearning {
    public:
        OnlineSMA() : lrate_(1e-1) {
            algo_name_ = "OnlineSMA";
        }

        OnlineSMA(double stop_thred, double lrate, double shrink_weight) :
                stop_thred_(stop_thred), lrate_(lrate), shrink_weight_(shrink_weight) {
            algo_name_ = "OnlineSMA";
        }

        virtual ~OnlineSMA() = default;

        void fit(std::vector<std::vector<double>> &trains,
                 std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, int num_iters=1, bool verbose = false) override;

    private:
        double lrate_;
        double shrink_weight_;
        double stop_thred_;
    };

    class OnlineExpectMax : public OnlineParamLearning {
    public:
        OnlineExpectMax() : stop_thred_(1e-4), lap_lambda_(1) {
            algo_name_ = "OnlineExpectMax";
        }

        OnlineExpectMax(double stop_thred, double lap_lambda) :
                stop_thred_(stop_thred), lap_lambda_(lap_lambda) {
            algo_name_ = "OnlineExpectMax";
        }

        virtual ~OnlineExpectMax() = default;

        void fit(std::vector<std::vector<double>> &trains,
                 std::vector<std::vector<double>> &valids,
                 SPNetwork &spn, int num_iters=1, bool verbose=false) override;

    protected:
        double stop_thred_;
        double lap_lambda_;
    };

    class OnlineCollapsedVB : public OnlineParamLearning {
    public:
        OnlineCollapsedVB() : stop_thred_(1e-3), lrate_(1e-1), prior_scale_(1e2),
                               seed_(42) {
            algo_name_ = "OnlineCollapsedVB";
        }

        OnlineCollapsedVB(double stop_thred, double lrate, double prior_scale,
                           uint seed=42) : stop_thred_(stop_thred), lrate_(lrate),
                                           prior_scale_(prior_scale), seed_(seed) {
            algo_name_ = "OnlineCollapsedVB";
        }

        virtual ~OnlineCollapsedVB() = default;

        void fit(std::vector<std::vector<double>> &trains,
                 std::vector<std::vector<double>> &valids, SPNetwork &spn,
                 int num_iters=1, bool verbose=false) override;

    private:
        double lrate_;
        double stop_thred_;
        double prior_scale_;
        uint seed_;
    };

    class OnlineADF : public OnlineParamLearning {
    public:
        OnlineADF() : stop_thred_(1e-3), prior_scale_(1e2) {
            algo_name_ = "OnlineADF";
        }

        OnlineADF(double stop_thred, double prior_scale) :
                stop_thred_(stop_thred), prior_scale_(prior_scale) {
            algo_name_ = "OnlineADF";
        }

        virtual ~OnlineADF() = default;

        void fit(std::vector<std::vector<double>> &trains,
                 std::vector<std::vector<double>> &valids, SPNetwork &spn,
                 int num_iters=1, bool verbose=false) override;

    private:
        double stop_thred_;
        double prior_scale_;
    };

    class OnlineBMM : public OnlineParamLearning {
    public:
        OnlineBMM() : stop_thred_(1e-3), prior_scale_(1e2) {
            algo_name_ = "OnlineBMM";
        }

        OnlineBMM(double stop_thred, double prior_scale) : stop_thred_(stop_thred), prior_scale_(prior_scale) {
            algo_name_ = "OnlineBMM";
        }

        virtual ~OnlineBMM() = default;

        void fit(std::vector<std::vector<double>> &trains,
                 std::vector<std::vector<double>> &valids, SPNetwork &spn,
                 int num_iters=1, bool verbose=false) override;

    private:
        double stop_thred_;
        double prior_scale_;
    };
}

#endif //SPN_EM_ONLINEPARAMLEARNING_H
