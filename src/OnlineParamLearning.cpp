//
// Created by Han Zhao on 7/13/15.
//

#include "OnlineParamLearning.h"

#include "utils.h"
#include <map>
#include <random>

namespace SPN {
    // Online Expectation Maximization algorithm for learning the model parameters of SPNs.
    // The implementation is consistent with the so called incremental EM as well as step-wise
    // EM where when decay_gamma = 1, it reduces to incremental EM and when 0 < decay_gamma < 1,
    // the implementation reduces to step-wise EM.
    void OnlineExpectMax::fit(std::vector<std::vector<double>> &trains,
                              std::vector<std::vector<double>> &valids,
                              SPN::SPNetwork &spn, int num_iters, bool verbose) {
        // Random number generators.
        std::random_device rd;
        std::mt19937 g(rd());
        size_t num_var = trains[0].size();
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        // Initialization
        double train_logps, valid_logps;
        double ssz;
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Initialize SST based on the structure of the network.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
            }
        }
        // Renormalize the weight before parameter learning.
        spn.weight_projection();
        // Start expectation maximization.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters; ++t) {
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            // Record current training and validation set log-likelihoods.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            train_logps /= num_trains;
            valid_logps /= num_valids;
            // Store statistics.
            train_funcs.push_back(train_logps);
            valid_funcs.push_back(valid_logps);
            // Update the optimal model weights.
            if (valid_logps > optimal_valid_logp) {
                optimal_valid_logp = valid_logps;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            opt[pt][k] = ((SumNode *) pt)->weights()[k];
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            // Clean previous record.
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            // Reshuffle the training vectors.
            std::shuffle(trains.begin(), trains.end(), g);
            for (size_t n = 0; n < num_trains; ++n) {
                // Bottom-up and top-down passes of the network.
                spn.EvalDiff(trains[n], mask_false);
                // Compute the first term of the gradient.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        // The first term of the projected gradient descent update formula.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sst[pt][k] += ((SumNode *) pt)->weights()[k] *
                                          exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                        }
                    }
                }
                // Parameter update using online expectation maximization.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        ssz = 0.0;
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            ssz += sst[pt][k] + lap_lambda_;
                        }
                        // Weight update.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            ((SumNode *) pt)->set_weight(k, (sst[pt][k] + lap_lambda_) / ssz);
                        }
                    }
                }
            }
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weight parameter encountered during optimization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->set_weight(k, opt[pt][k]);
                }
            }
        }
    }

    // Online learning of Sum-Product Networks using exponentiated gradient descent algorithm.
    void OnlineExpoGD::fit(std::vector<std::vector<double>> &trains,
                           std::vector<std::vector<double>> &valids,
                           SPNetwork &spn, int num_iters, bool verbose) {
        // Random number generators.
        std::random_device rd;
        std::mt19937 g(rd());
        // Initialization of all the model parameters.
        size_t num_var = trains[0].size();
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        std::vector<bool> mask_true(num_var, true);
        std::vector<double> all_one(num_var, 1.0);
        double train_logps, valid_logps;
        double original_weight, new_weight;
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Local learning rate.
        double lrate = lrate_;
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Initialize SST based on the structure of the network.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
            }
        }
        // Start exponentiated gradient descent.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters; ++t) {
            // Not good fitting, shrinking the weight.
            if (t > 1 && train_funcs[t-1] < train_funcs[t-2]) {
                lrate *= shrink_weight_;
            }
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            // Record current training and validation set log-likelihoods.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            // Propagate value of all one vector.
            spn.EvalDiff(all_one, mask_true);
            train_logps -= num_trains * spn.root_->fr();
            valid_logps -= num_valids * spn.root_->fr();
            train_logps /= num_trains;
            valid_logps /= num_valids;
            // Store statistics.
            train_funcs.push_back(train_logps);
            valid_funcs.push_back(valid_logps);
            // Update the optimal model weights.
            if (valid_logps > optimal_valid_logp) {
                optimal_valid_logp = valid_logps;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            opt[pt][k] = ((SumNode *) pt)->weights()[k];
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            // Online training.
            // Reshuffle training vectors.
            std::shuffle(trains.begin(), trains.end(), g);
            for (size_t n = 0; n < num_trains; ++n) {
                // Clean previous record.
                for (auto &pvec : sst) {
                    for (size_t k = 0; k < pvec.second.size(); ++k) {
                        pvec.second[k] = 0.0;
                    }
                }
                // Bottom-up and top-down passes of the network.
                spn.EvalDiff(trains[n], mask_false);
                // Compute the first term of the gradient.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        // The first term of the projected gradient descent update formula.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sst[pt][k] += exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
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
                            sst[pt][k] -= exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                        }
                    }
                }
                // Parameter update using projected gradient descent.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            original_weight = ((SumNode *) pt)->weights()[k];
                            new_weight = original_weight * exp(lrate * sst[pt][k]);
                            ((SumNode *) pt)->set_weight(k, new_weight);
                        }
                    }
                }
            }
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weight parameter encountered during optimization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->set_weight(k, opt[pt][k]);
                }
            }
        }
        // Renormalize the weight after parameter learning.
        spn.weight_projection(1e-3);
    }

    void OnlineProjectedGD::fit(std::vector<std::vector<double>> &trains,
                                std::vector<std::vector<double>> &valids,
                                SPNetwork &spn, int num_iters, bool verbose) {
        // Random number generators.
        std::random_device rd;
        std::mt19937 g(rd());
        // Initialization of all the modle parameters.
        size_t num_var = trains[0].size();
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        std::vector<bool> mask_true(num_var, true);
        std::vector<double> all_one(num_var, 1.0);
        double train_logps, valid_logps;
        double original_weight, new_weight;
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Local learning rate.
        double lrate = lrate_;
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Initialize SST based on the structure of the network.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
            }
        }
        // Start projected gradient descent.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters; ++t) {
            // Not good fitting, shrinking the weight.
            if (t > 1 && train_funcs[t-1] < train_funcs[t-2]) {
                lrate *= shrink_weight_;
            }
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            // Record current training and validation set log-likelihoods.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            // Propagate value of all one vector.
            spn.EvalDiff(all_one, mask_true);
            train_logps -= num_trains * spn.root_->fr();
            valid_logps -= num_valids * spn.root_->fr();
            train_logps /= num_trains;
            valid_logps /= num_valids;
            // Store statistics.
            train_funcs.push_back(train_logps);
            valid_funcs.push_back(valid_logps);
            // Update the optimal model weights.
            if (valid_logps > optimal_valid_logp) {
                optimal_valid_logp = valid_logps;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            opt[pt][k] = ((SumNode *) pt)->weights()[k];
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            // Online training.
            // Reshuffle training vectors.
            std::shuffle(trains.begin(), trains.end(), g);
            for (size_t n = 0; n < num_trains; ++n) {
                // Clean previous record.
                for (auto &pvec : sst) {
                    for (size_t k = 0; k < pvec.second.size(); ++k) {
                        pvec.second[k] = 0.0;
                    }
                }
                // Bottom-up and top-down passes of the network.
                spn.EvalDiff(trains[n], mask_false);
                // Compute the first term of the gradient.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        // The first term of the projected gradient descent update formula.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sst[pt][k] += exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
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
                            sst[pt][k] -= exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                        }
                    }
                }
                // Parameter update using projected gradient descent.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            original_weight = ((SumNode *) pt)->weights()[k];
                            new_weight = original_weight + lrate * sst[pt][k] > 0 ?
                                         original_weight + lrate * sst[pt][k] : proj_eps_;
                            ((SumNode *) pt)->set_weight(k, new_weight);
                        }
                    }
                }
            }
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weight parameter encountered during optimization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->set_weight(k, opt[pt][k]);
                }
            }
        }
        // Renormalize the weight after parameter learning.
        spn.weight_projection(1e-3);
    }

    void OnlineSMA::fit(std::vector<std::vector<double>> &trains,
                        std::vector<std::vector<double>> &valids,
                        SPNetwork &spn, int num_iters, bool verbose) {
        // Initialization of all the modle parameters.
        size_t num_var = trains[0].size();
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        std::vector<bool> mask_true(num_var, true);
        std::vector<double> all_one(num_var, 1.0);
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        double train_logps, valid_logps;
        double original_weight, new_weight;
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Local learning rate.
        double lrate = lrate_;
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Initialize SST based on the structure of the network.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
            }
        }
        // Start projected gradient descent.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters; ++t) {
            // Not good fitting, shrinking the weight.
            if (t > 1 && train_funcs[t-1] < train_funcs[t-2]) {
                lrate *= shrink_weight_;
            }
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            // Record current training and validation set log-likelihoods.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            // Propagate value of all one vector.
            spn.EvalDiff(all_one, mask_true);
            train_logps -= num_trains * spn.root_->fr();
            valid_logps -= num_valids * spn.root_->fr();
            train_logps /= num_trains;
            valid_logps /= num_valids;
            // Store statistics.
            train_funcs.push_back(train_logps);
            valid_funcs.push_back(valid_logps);
            // Update the optimal model weights.
            if (valid_logps > optimal_valid_logp) {
                optimal_valid_logp = valid_logps;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            opt[pt][k] = ((SumNode *) pt)->weights()[k];
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            // Online training.
            for (size_t n = 0; n < num_trains; ++n) {
                // Clean previous record.
                for (auto &pvec : sst) {
                    for (size_t k = 0; k < pvec.second.size(); ++k) {
                        pvec.second[k] = 0.0;
                    }
                }
                // Bottom-up and top-down passes of the network.
                spn.EvalDiff(trains[n], mask_false);
                // Compute the first term of the gradient.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        // The first term of the projected gradient descent update formula.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sst[pt][k] += ((SumNode *) pt)->weights()[k] *
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
                            sst[pt][k] -= ((SumNode *) pt)->weights()[k] *
                                    exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                        }
                    }
                }
                // Parameter update using online sequential monomial approxiamtion.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            original_weight = ((SumNode *) pt)->weights()[k];
                            new_weight = original_weight * exp(lrate * sst[pt][k]);
                            ((SumNode *) pt)->set_weight(k, new_weight);
                        }
                    }
                }
            }
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weight parameter encountered during optimization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->set_weight(k, opt[pt][k]);
                }
            }
        }
        // Renormalize the weight after parameter learning.
        spn.weight_projection(1e-3);
    }

    void OnlineCollapsedVB::fit(std::vector<std::vector<double>> &trains,
                                std::vector<std::vector<double>> &valids,
                                SPNetwork &spn, int num_iters, bool verbose) {
        // Random number generators.
        std::random_device rd;
        std::mt19937 g(rd());
        // Initialization
        size_t num_var = trains[0].size();
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        double train_logps, valid_logps;
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Hyperparameters.
        double lrate = lrate_;
        double prior_scale = prior_scale_;
        double fudge_factor = 1e-2;
        double sum_beta = 0.0, sum_alpha = 0.0, kl_cost = 0.0;
        // Prior variational parameter alpha.
        std::map<SPNNode *, std::vector<double>> alpha;
        for (SPNNode *pt : spn.top_down_order()) {
            auto sum_pt = dynamic_cast<SumNode*>(pt);
            if (sum_pt) {
                // Initialize SST based on the structure of the network.
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
                // Initialize the prior alpha_k based on the weights.
                auto alpha_k = sum_pt->weights();
                std::for_each(alpha_k.begin(), alpha_k.end(),
                              [prior_scale](double& d) {d *= prior_scale;});
                alpha.insert({pt, alpha_k});
                // Initialize the posterior beta_k = alpha_k.
                sum_pt->values_ = alpha_k;
            }
        }
        // Random initialization.
        spn.set_random_params(seed_);
        // Online Collapsed Variational Bayesian inference.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters; ++t) {
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            // Record current training and validation set log-likelihoods.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            train_logps /= num_trains;
            valid_logps /= num_valids;
            // Store statistics.
            train_funcs.push_back(train_logps);
            valid_funcs.push_back(valid_logps);
            // Update the optimal model weights.
            if (valid_logps > optimal_valid_logp) {
                optimal_valid_logp = valid_logps;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            opt[pt][k] = ((SumNode *) pt)->weights()[k];
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            // Clean previous record.
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            // Reshuffle the training vectors.
            std::shuffle(trains.begin(), trains.end(), g);
            for (size_t n = 0; n < num_trains; ++n) {
                // Bottom-up evaluation and top-down differentiation of the network.
                spn.EvalDiff(trains[n], mask_false);
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        auto sum_pt = (SumNode*) pt;
                        sum_alpha = 0.0;
                        sum_beta = 0.0;
                        for (auto v : sum_pt->values_) sum_beta += v;
                        for (auto v : alpha[pt]) sum_alpha += v;
                        // Compute the gradient of the data-fitting term.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sst[pt][k] += ((SumNode *) pt)->weights()[k] *
                                          exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                        }
                        // Compute the gradient of the KL-div between the variational posterior and the prior.
                        // Use psi'(x) ~ 1/x to approximate the Trigamma function when x > 0.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            kl_cost = 0.0;
                            kl_cost -= (sum_pt->values_[k] - alpha[pt][k]) / sum_pt->values_[k];
                            kl_cost += ((sum_beta - sum_alpha) - (sum_pt->values_[k] - alpha[pt][k])) / sum_beta;
                            kl_cost /= num_trains;
                            sst[pt][k] += kl_cost;
                        }
                    }
                }
                // Parameter updating to the posterior mean of the collapsed variational distribution.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        auto sum_pt = (SumNode*) pt;
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sum_pt->values_[k] += lrate * sst[pt][k];
                            sum_pt->values_[k] = sum_pt->values_[k] >= 0.0 ? sum_pt->values_[k] : fudge_factor;
                        }
                        // Update model parameter using the variational posterior mean estimator.
                        sum_beta = 0.0;
                        for (auto v : sum_pt->values_) sum_beta += v;
                        auto weights = sum_pt->values_;
                        std::for_each(weights.begin(), weights.end(), [sum_beta](double& d) {d /= sum_beta;});
                        sum_pt->set_weights(weights);
                    }
                }
            }
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weight parameter encountered during optimization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->set_weight(k, opt[pt][k]);
                }
            }
        }
    }

    void OnlineADF::fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
                        SPNetwork &spn, int num_iters, bool verbose) {
        // Random number generators.
        std::random_device rd;
        std::mt19937 g(rd());
        // Initialization
        size_t num_var = trains[0].size();
        size_t num_children = 0;
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        double train_logps, valid_logps;
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // lambda statistics for update in each iteration, i.e., lambda = w_k,j x V_j x D_k.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Hyperparameters.
        double prior_scale = prior_scale_;
        double sum_alpha = 0.0;
        double fudge_factor = 1e-2;
        for (SPNNode *pt : spn.top_down_order()) {
            auto sum_pt = dynamic_cast<SumNode*>(pt);
            if (sum_pt) {
                // Initialize SST based on the structure of the network.
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
                // Initialize the prior alpha_k uniformly.
                auto alpha_k = sum_pt->weights();
                num_children = alpha_k.size();
                std::for_each(alpha_k.begin(), alpha_k.end(),
                              [num_children](double& d) {d = 1.0 / num_children;});
                // Initialize the posterior beta_k = alpha_k.
                sum_pt->values_ = alpha_k;
            }
        }
        // Online assumed density filtering.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters; ++t) {
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            // Record current training and validation set log-likelihoods.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            train_logps /= num_trains;
            valid_logps /= num_valids;
            // Store statistics.
            train_funcs.push_back(train_logps);
            valid_funcs.push_back(valid_logps);
            // Update the optimal model weights.
            if (valid_logps > optimal_valid_logp) {
                optimal_valid_logp = valid_logps;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            opt[pt][k] = ((SumNode *) pt)->weights()[k];
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            // Clean previous record.
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            // Reshuffle the training vectors.
            std::shuffle(trains.begin(), trains.end(), g);
            for (size_t n = 0; n < num_trains; ++n) {
                // Bottom-up evaluation and top-down differentiation of the network.
                spn.EvalDiff(trains[n], mask_false);
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        auto sum_pt = (SumNode*) pt;
                        // Compute the lambda statistics in the paper.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sst[pt][k] = sum_pt->weights()[k] *
                                    exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            assert (sst[pt][k] >= 0.0 && sst[pt][k] <= 1.0);
                        }
                    }
                }
                // Parameter updating to the posterior mean of the assumed density filtering,
                // in this case moment matching reduces to update using weighted geometric mean.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        auto sum_pt = (SumNode*) pt;
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sum_pt->values_[k] = 0.0;
                            sum_pt->values_[k] += (1.0 - sst[pt][k]) *
                                    log(std::max(fudge_factor, (prior_scale * sum_pt->weights()[k] - 0.5)) /
                                                (prior_scale - 0.5));
                            sum_pt->values_[k] += sst[pt][k] * log((prior_scale * sum_pt->weights()[k] + 0.5)
                                                                   / (prior_scale + 0.5));
                            sum_pt->values_[k] = (prior_scale - 0.5) * exp(sum_pt->values_[k]) + 0.5;
                        }
                        // Update model parameter using the posterior mean of the one-step update posterior.
                        sum_alpha = 0.0;
                        for (auto v : sum_pt->values_) sum_alpha += v;
                        auto weights = sum_pt->values_;
                        std::for_each(weights.begin(), weights.end(), [sum_alpha](double& d) {d /= sum_alpha;});
                        sum_pt->set_weights(weights);
                    }
                }
            }
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weight parameter encountered during optimization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->set_weight(k, opt[pt][k]);
                }
            }
        }
    }

    void OnlineBMM::fit(std::vector<std::vector<double>> &trains, std::vector<std::vector<double>> &valids,
                        SPNetwork &spn, int num_iters, bool verbose) {
        // Random number generators.
        std::random_device rd;
        std::mt19937 g(rd());
        // Initialization
        size_t num_var = trains[0].size();
        size_t num_children = 0;
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        double train_logps, valid_logps;
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // lambda statistics for update in each iteration, i.e., lambda = w_k,j x V_j x D_k.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Hyperparameters.
        double prior_scale = prior_scale_;
        double sum_alpha = 0.0;
        double fudge_factor = 1e-2;
        for (SPNNode *pt : spn.top_down_order()) {
            auto sum_pt = dynamic_cast<SumNode*>(pt);
            if (sum_pt) {
                // Initialize SST based on the structure of the network.
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
                // Initialize the prior alpha_k uniformly.
                auto alpha_k = sum_pt->weights();
                num_children = alpha_k.size();
                std::for_each(alpha_k.begin(), alpha_k.end(),
                              [num_children](double& d) {d = 1.0 / num_children;});
                // Initialize the posterior beta_k = alpha_k.
                sum_pt->values_ = alpha_k;
            }
        }
        // Online assumed density filtering.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters; ++t) {
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            // Record current training and validation set log-likelihoods.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            train_logps /= num_trains;
            valid_logps /= num_valids;
            // Store statistics.
            train_funcs.push_back(train_logps);
            valid_funcs.push_back(valid_logps);
            // Update the optimal model weights.
            if (valid_logps > optimal_valid_logp) {
                optimal_valid_logp = valid_logps;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            opt[pt][k] = ((SumNode *) pt)->weights()[k];
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            // Clean previous record.
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            // Reshuffle the training vectors.
            std::shuffle(trains.begin(), trains.end(), g);
            for (size_t n = 0; n < num_trains; ++n) {
                // Bottom-up evaluation and top-down differentiation of the network.
                spn.EvalDiff(trains[n], mask_false);
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        auto sum_pt = (SumNode*) pt;
                        // Compute the lambda statistics in the paper.
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sst[pt][k] = sum_pt->weights()[k] *
                                         exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            assert (sst[pt][k] >= 0.0 && sst[pt][k] <= 1.0);
                        }
                    }
                }
                // Parameter updating to the posterior mean of the Bayesian moment matching posterior.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        auto sum_pt = (SumNode*) pt;
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            sum_pt->values_[k] = (1.0 - sst[pt][k]) * sum_pt->weights()[k] +
                                    sst[pt][k] * (prior_scale * sum_pt->weights()[k] + 1) / (prior_scale + 1);
                        }
                        // Update model parameter using the posterior mean of the one-step update posterior.
                        sum_alpha = 0.0;
                        for (auto v : sum_pt->values_) sum_alpha += v;
                        auto weights = sum_pt->values_;
                        std::for_each(weights.begin(), weights.end(), [sum_alpha](double& d) {d /= sum_alpha;});
                        sum_pt->set_weights(weights);
                    }
                }
            }
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weight parameter encountered during optimization.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                for (size_t k = 0; k < pt->num_children(); ++k) {
                    ((SumNode *) pt)->set_weight(k, opt[pt][k]);
                }
            }
        }
    }
}
