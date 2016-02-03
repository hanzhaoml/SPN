//
// Created by Han Zhao on 11/18/15.
//

#include "StreamParamLearning.h"

#include <random>

namespace SPN {
    void StreamProjectedGD::fit(std::vector<double> &train, SPNetwork &spn, bool verbose) {
        // Masks for inference.
        std::vector<bool> mask_false(train.size(), false);
        std::vector<bool> mask_true(train.size(), true);
        std::vector<double> all_one(train.size(), 1.0);
        double original_weight = 0.0, new_weight = 0.0;
        // Streaming update.
        // Bottom-up and top-down passes of the network.
        spn.EvalDiff(train, mask_false);
        // Compute the first term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The first term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] = exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Bottom-up and top-down passes of the all-one vector.
        spn.EvalDiff(all_one, mask_true);
        // Compute the second term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The second term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] -= exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Parameter update using projected gradient descent.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    original_weight = ((SumNode *) pt)->weights()[k];
                    new_weight = original_weight + lrate_ * ((SumNode *) pt)->values_[k] > 0 ?
                                 original_weight + lrate_ * ((SumNode *) pt)->values_[k] : proj_eps_;
                    ((SumNode *) pt)->set_weight(k, new_weight);
                }
            }
        }
    }

    void StreamExpoGD::fit(std::vector<double> &train, SPNetwork &spn, bool verbose) {
        // Masks for inference.
        std::vector<bool> mask_false(train.size(), false);
        std::vector<bool> mask_true(train.size(), true);
        std::vector<double> all_one(train.size(), 1.0);
        double original_weight = 0.0, new_weight = 0.0;
        // Streaming update.
        // Bottom-up and top-down passes of the network.
        spn.EvalDiff(train, mask_false);
        // Compute the first term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The first term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] = exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Bottom-up and top-down passes of the all-one vector.
        spn.EvalDiff(all_one, mask_true);
        // Compute the second term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The second term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] -= exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Parameter update using exponentiated gradient descent.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    original_weight = ((SumNode *) pt)->weights()[k];
                    new_weight = original_weight * exp(lrate_ * ((SumNode *) pt)->values_[k]);
                    ((SumNode *) pt)->set_weight(k, new_weight);
                }
            }
        }
    }

    void StreamSMA::fit(std::vector<double> &train, SPNetwork &spn, bool verbose) {
        // Masks for inference.
        std::vector<bool> mask_false(train.size(), false);
        std::vector<bool> mask_true(train.size(), true);
        std::vector<double> all_one(train.size(), 1.0);
        double original_weight = 0.0, new_weight = 0.0;
        // Streaming update.
        // Bottom-up and top-down passes of the network.
        spn.EvalDiff(train, mask_false);
        // Compute the first term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The first term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] = ((SumNode *) pt)->weights()[k] *
                            exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Bottom-up and top-down passes of the all-one vector.
        spn.EvalDiff(all_one, mask_true);
        // Compute the second term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The second term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] -= ((SumNode *) pt)->weights()[k] *
                            exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Parameter update using sequential monomial approximation.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    original_weight = ((SumNode *) pt)->weights()[k];
                    new_weight = original_weight * exp(lrate_ * ((SumNode *) pt)->values_[k]);
                    ((SumNode *) pt)->set_weight(k, new_weight);
                }
            }
        }
    }

    void StreamExpectMax::fit(std::vector<double> &train, SPNetwork &spn, bool verbose) {
        // Masks for inference.
        std::vector<bool> mask_false(train.size(), false);
        std::vector<bool> mask_true(train.size(), true);
        std::vector<double> all_one(train.size(), 1.0);
        double original_weight = 0.0, new_weight = 0.0;
        // Streaming update.
        // Bottom-up and top-down passes of the network.
        spn.EvalDiff(train, mask_false);
        // Compute the first term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The first term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] = ((SumNode *) pt)->weights()[k] *
                            exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Bottom-up and top-down passes of the all-one vector.
        spn.EvalDiff(all_one, mask_true);
        // Compute the second term of the gradient.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // The second term of the projected gradient descent update formula.
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->values_[k] -= ((SumNode *) pt)->weights()[k] *
                                                    exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                }
            }
        }
        // Parameter update using incremental expectation maximization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    original_weight = ((SumNode *) pt)->weights()[k];
                    new_weight = original_weight + lrate_ * ((SumNode *) pt)->values_[k];
                    ((SumNode *) pt)->set_weight(k, new_weight);
                }
            }
        }
    }
}