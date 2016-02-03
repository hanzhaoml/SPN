//
// Created by Han Zhao on 11/18/15.
//

#ifndef SPN_EM_STREAMPARAMLEARNING_H
#define SPN_EM_STREAMPARAMLEARNING_H

#include "SPNetwork.h"

#include <vector>
#include <map>

namespace SPN {
    class StreamParamLearning {
    public:
        StreamParamLearning() = default;
        virtual ~StreamParamLearning() = default;

        std::string algo_name() const {
            return algo_name_;
        }
        // Streaming parameter learning algorithms to be implemented by subclasses.
        virtual void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) { };

    protected:
        std::string algo_name_ = "AbstractStreamParamLearning";
    };

    // Streaming projected gradient descent for learning the parameters of Sum-Product Networks.
    class StreamProjectedGD : public StreamParamLearning {
    public:
        StreamProjectedGD() : proj_eps_(1e-2), lrate_(1e-1) {
            algo_name_ = "StreamProjectedGD";
        }
        StreamProjectedGD(double proj_eps, double lrate) : proj_eps_(proj_eps), lrate_(lrate) {
            algo_name_ = "StreamProjectedGD";
        }

        virtual ~StreamProjectedGD() = default;

        void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override;

    protected:
        double lrate_;
        double proj_eps_;
    };

    // Streaming exponentiated gradient descent for leanring the parameters of Sum-Product Networks.
    class StreamExpoGD : public StreamParamLearning {
    public:
        StreamExpoGD() : lrate_(1e-1) {
            algo_name_ = "StreamExpoGD";
        }
        StreamExpoGD(double lrate) : lrate_(lrate) {
            algo_name_ = "StreamExpoGD";
        }

        virtual ~StreamExpoGD() = default;

        void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override;

    protected:
        double lrate_;
    };

    // Streaming sequential monomial minimization algorithm for learning the parameters of Sum-Product Networks.
    class StreamSMA : public StreamParamLearning {
    public:
        StreamSMA() : lrate_(1e-1) {
            algo_name_ = "StreamSMA";
        }
        StreamSMA(double lrate) : lrate_(lrate) {
            algo_name_ = "StreamSMA";
        }

        virtual ~StreamSMA() = default;

        void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override;

    protected:
        double lrate_;
    };

    // Streaming expectation maximization for learning the parameters of Sum-Product Networks.
    // In the streaming learning setting, incremental EM update is not applicable, only the stepwise
    // version is applicable because we cannot keep a record for each of the instance.
    class StreamExpectMax : public StreamParamLearning {
    public:
        StreamExpectMax() : lrate_(1.0) {
            algo_name_ = "StreamExpectMax";
        }

        StreamExpectMax(double lrate) : lrate_(lrate) {
            algo_name_ = "StreamExpectMax";
        }

        virtual ~StreamExpectMax() = default;

        void fit(std::vector<double> &train, SPNetwork &spn, bool verbose = false) override;

    protected:
        double lrate_;
    };
}


#endif //SPN_EM_STREAMPARAMLEARNING_H
