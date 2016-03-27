//
// Created by Han Zhao on 5/31/15.
//

#include "BatchParamLearning.h"
#include <map>
#include <queue>
#include <stack>


namespace SPN {
    // Batch projected gradient descent algorithm for learning the model parameters of SPNs.
    // Note that at each iteration we need to project the intermediate solution to an
    // epsilon-orthant, not the nonnegative orthant in order to avoid all the components of
    // model to be 0.
    void ProjectedGD::fit(const std::vector<std::vector<double>> &trains,
                          const std::vector<std::vector<double>> &valids,
                          SPNetwork &spn, bool verbose) {
        // Initialization
        size_t num_var = trains[0].size();
        std::vector<double> all_one(num_var, 1.0);
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        double train_logps, valid_logps;
        double original_weight, new_weight;
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Local learning rate.
        double lrate = lrate_;
        // Sufficient statistics for update in each iteration
        // and optimal weights enconutered during optimization.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        std::vector<bool> mask_true(num_var, true);
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
        for (size_t t = 0; t < num_iters_; ++t) {
            // Not good fitting, shrinking the weight.
            if (t > 1 && train_funcs[t-1] < train_funcs[t-2]) {
                lrate *= shrink_weight_;
            }
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                // Bottom-up and top-down passes of the network.
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                    // Compute gradients.
                    for (SPNNode *pt : spn.top_down_order()) {
                        if (pt->type() == SPNNodeType::SUMNODE) {
                            // Projected gradient descent update formula.
                            for (size_t k = 0; k < pt->num_children(); ++k) {
                                sst[pt][k] += exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            }
                        }
                    }
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            // Propagate value of all one vector.
            spn.EvalDiff(all_one, mask_true);
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        sst[pt][k] -= num_trains * exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                    }
                }
            }
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
            // Weight update using projected gradient descent.
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        sst[pt][k] /= num_trains;
                        original_weight = ((SumNode *) pt)->weights()[k];
                        new_weight = original_weight + lrate * sst[pt][k] > 0 ?
                                     original_weight + lrate * sst[pt][k] : proj_eps_;
                        ((SumNode *) pt)->set_weight(k, new_weight);
                    }
                }
            }
            // Stop criterion.
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weights during the optization.
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

    // Batch Expectation Maximization algorithm for learning the model parameters of SPNs
    void ExpectMax::fit(const std::vector<std::vector<double>> &trains,
                        const std::vector<std::vector<double>> &valids,
                        SPNetwork &spn, bool verbose) {
        // Initialization
        double train_logps, valid_logps;
        double iter_mean, iter_var;
        size_t num_var = trains[0].size();
        double ssz;
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        // Optimal log-likelihood on the validation set.
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        double epsilon = std::numeric_limits<double>::epsilon();
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, vst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Initialize SST based on the structure of the network.
        // Optimal model weights achieved during optimization based on the
        // validation set log-likelihoods.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
            }
        }
        // Initialize VST based on the number of parameters of each leaf
        // univariate distribution.
        for (VarNode *pt : spn.dist_nodes()) {
            vst.insert({pt, std::vector<double>(pt->num_param() + 1)});
            opt.insert({pt, std::vector<double>(pt->num_param())});
        }
        // Start expectation maximization.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters_; ++t) {
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            for (auto &pvec : vst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                // Bottom-up and top-down passes of the network.
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                    // Compute sufficient statistics.
                    for (SPNNode *pt : spn.top_down_order()) {
                        if (pt->type() == SPNNodeType::SUMNODE) {
                            for (size_t k = 0; k < pt->num_children(); ++k) {
                                sst[pt][k] += ((SumNode *) pt)->weights()[k] *
                                        exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            }
                        } else if (pt->type() == SPNNodeType::VARNODE) {
                            // Normal leaf univariate distributions.
                            if (((VarNode *) pt)->distribution() == VarNodeType::NORMALNODE) {
                                vst[pt][0] += exp(pt->dr() + pt->fr() - spn.root_->fr());
                                vst[pt][1] += trains[n][((VarNode *) pt)->var_index()] *
                                        exp(pt->dr() + pt->fr() - spn.root_->fr());
                                vst[pt][2] += pow(trains[n][((VarNode *) pt)->var_index()], 2) *
                                        exp(pt->dr() + pt->fr() - spn.root_->fr());
                            }
                        }
                    }
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
                    } else if (pt->type() == SPNNodeType::VARNODE) {
                        // Normal leaf univariate distributions have two sufficient statistics.
                        if (((VarNode *) pt)->distribution() == VarNodeType::NORMALNODE) {
                            opt[pt][0] = ((NormalNode *) pt)->var_mean();
                            opt[pt][1] = ((NormalNode *) pt)->var_var();
                        }
                    }
                }
            }
            if (verbose) {
                std::cout << t << "," << train_funcs[t] << "," << valid_funcs[t] << std::endl;
            }
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    ssz = 0.0;
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        // Expected sufficient statistics (pseudo count + laplacian smoothing)
                        ssz += sst[pt][k] + epsilon;
                    }
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        ((SumNode *) pt)->set_weight(k, (sst[pt][k] + epsilon) / ssz);
                    }
                } else if (pt->type() == SPNNodeType::VARNODE) {
                    // Normal leaf univariate distributions.
                    if (((VarNode *) pt)->distribution() == VarNodeType::NORMALNODE) {
                        iter_mean = vst[pt][1] / vst[pt][0];
                        iter_var = vst[pt][2] / vst[pt][0] - iter_mean * iter_mean;
                        iter_var = iter_var > 0 ? iter_var : epsilon;
                        ((NormalNode *) pt)->set_var_mean(iter_mean);
                        ((NormalNode *) pt)->set_var_var(iter_var);
                    }
                }
            }
            // Stop criterion.
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
            } else if (pt->type() == SPNNodeType::VARNODE) {
                // Normal leaf univariate distributions.
                if (((VarNode *) pt)->distribution() == VarNodeType::NORMALNODE) {
                    ((NormalNode *) pt)->set_var_mean(opt[pt][0]);
                    ((NormalNode *) pt)->set_var_var(opt[pt][1]);
                }
            }
        }
    }

    // Batch Collapsed Variational Bayes inference algorithm.
    void CollapsedVB::fit(const std::vector<std::vector<double>> &trains,
                          const std::vector<std::vector<double>> &valids,
                          SPNetwork &spn, bool verbose) {
        // Initialization
        size_t num_var = trains[0].size();
        double train_logps, valid_logps;
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        double lrate = lrate_;
        double prior_scale = prior_scale_;
        double fudge_factor = 1e-2;
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        // Optimal log-likelihood on the validation set.
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Prior variational parameter alpha.
        std::map<SPNNode *, std::vector<double>> alpha;
        for (SPNNode *pt : spn.top_down_order()) {
            auto sum_pt = dynamic_cast<SumNode*>(pt);
            if (sum_pt) {
                // Initialize SST based on the structure of the network.
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
                // Initialize the prior alpha_k = 100 * weight_k
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
        // Start collapsed variational bayes expectation maximization.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters_; ++t) {
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                // Bottom-up and top-down passes of the network.
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                    // Compute sufficient statistics.
                    for (SPNNode *pt : spn.top_down_order()) {
                        if (pt->type() == SPNNodeType::SUMNODE) {
                            for (size_t k = 0; k < pt->num_children(); ++k) {
                                sst[pt][k] += ((SumNode *) pt)->weights()[k] *
                                              exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            }
                        }
                    }
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
            // Update optimization variables and also model parameters.
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    auto sum_pt = (SumNode*) pt;
                    double sum_beta = 0.0, sum_alpha = 0.0;
                    for (auto v : sum_pt->values_) sum_beta += v;
                    for (auto v : alpha[pt]) sum_alpha += v;
                    // Use psi'(x) ~ 1/x to approximate the Trigamma function when x > 0.
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        // The gradient of the KL-loss part between the variational posterior and the prior.
                        sst[pt][k] -= (sum_pt->values_[k] - alpha[pt][k]) / sum_pt->values_[k];
                        sst[pt][k] += ((sum_beta - sum_alpha) - (sum_pt->values_[k] - alpha[pt][k])) / sum_beta;
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
            // Stop criterion.
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

    // Exponentiated gradient method for optimizing the model parameters. Using multiplicative
    // updates computed from the log-likelihood function.
    void ExpoGD::fit(const std::vector<std::vector<double>> &trains,
                     const std::vector<std::vector<double>> &valids,
                     SPNetwork &spn, bool verbose) {
        // Initialization
        size_t num_var = trains[0].size();
        std::vector<double> all_one(num_var, 1.0);
        // Optimal validation set log-likelihood during optimization.
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        // Initialization
        double train_logps, valid_logps;
        double original_weight, new_weight;
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Local learning rate.
        double lrate = lrate_;
        // Sufficient statistics for update in each iteration.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        std::vector<bool> mask_true(num_var, true);
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
        for (size_t t = 0; t < num_iters_; ++t) {
            // Not good fitting, shrinking weight.
            if (t > 1 && train_funcs[t-1] < train_funcs[t-2]) {
                lrate *= shrink_weight_;
            }
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                // Bottom-up and top-down passes of the network.
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                    // Compute gradients.
                    for (SPNNode *pt : spn.top_down_order()) {
                        if (pt->type() == SPNNodeType::SUMNODE) {
                            // Projected gradient descent update formula.
                            for (size_t k = 0; k < pt->num_children(); ++k) {
                                sst[pt][k] += exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            }
                        }
                    }
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            // Propagate value of all one vector.
            spn.EvalDiff(all_one, mask_true);
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        sst[pt][k] -= num_trains * exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                    }
                }
            }
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
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        sst[pt][k] /= num_trains;
                        original_weight = ((SumNode *) pt)->weights()[k];
                        new_weight = original_weight * exp(lrate * sst[pt][k]);
                        ((SumNode *) pt)->set_weight(k, new_weight);
                    }
                }
            }
            // Stop criterion.
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
        // Re-normalize each sum node.
        spn.weight_projection(1e-3);
    }

    void SMA::fit(const std::vector<std::vector<double>> &trains,
                  const std::vector<std::vector<double>> &valids,
                  SPNetwork &spn, bool verbose) {
        // Initialization
        size_t num_var = trains[0].size();
        std::vector<double> all_one(num_var, 1.0);
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        std::vector<bool> mask_true(num_var, true);
        // Optimal validation set log-likelihood.
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        // Initialization
        double train_logps, valid_logps;
        double original_weight, new_weight;
        double lrate = lrate_;
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
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        for (size_t t = 0; t < num_iters_; ++t) {
            // Not good fitting, shrinking the weight.
            if (t > 1 && train_funcs[t-1] < train_funcs[t-2]) {
                lrate *= shrink_weight_;
            }
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                // Bottom-up and top-down passes of the network.
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                    // Compute sufficient statistics.
                    for (SPNNode *pt : spn.top_down_order()) {
                        if (pt->type() == SPNNodeType::SUMNODE) {
                            // Projected gradient descent update formula.
                            for (size_t k = 0; k < pt->num_children(); ++k) {
                                sst[pt][k] += ((SumNode *) pt)->weights()[k] *
                                        exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            }
                        }
                    }
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            // Propagate value of all one vector.
            spn.EvalDiff(all_one, mask_true);
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        sst[pt][k] -= num_trains * ((SumNode *) pt)->weights()[k] *
                                exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                    }
                }
            }
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
            // Weight update using projected gradient descent.
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        sst[pt][k] /= num_trains;
                        original_weight = ((SumNode *) pt)->weights()[k];
                        new_weight = original_weight * exp(lrate * sst[pt][k]);
                        ((SumNode *) pt)->set_weight(k, new_weight);
                    }
                }
            }
            // Stop criterion.
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
        // Re-normalize each sum node.
        spn.weight_projection(1e-3);
    }

    void LBFGS::fit(const std::vector<std::vector<double>> &trains, const std::vector<std::vector<double>> &valids,
                    SPNetwork &spn, bool verbose) {
        // Initialization
        size_t num_var = trains[0].size();
        std::vector<double> all_one(num_var, 1.0);
        double optimal_valid_logp = -std::numeric_limits<double>::infinity();
        double train_logps, valid_logps;
        double original_weight, new_weight;
        size_t num_trains = trains.size();
        size_t num_valids = valids.size();
        // Local learning rate.
        double lrate = lrate_;
        // Sufficient statistics for update in each iteration
        // and optimal weights encountered during optimization.
        std::map<SPNNode *, std::vector<double>> sst, opt;
        // History parameter vectors and gradient vectors.
        std::vector<std::map<SPNNode *, std::vector<double>>> history_grads(history_window_);
        std::vector<std::map<SPNNode *, std::vector<double>>> history_param(history_window_);
        std::vector<double> history_alphas(history_window_, 0.0);
        std::vector<double> history_phos(history_window_, 0.0);
        // Previous parameter vector and gradient vector.
        std::map<SPNNode *, std::vector<double>> prev_grads;
        std::map<SPNNode *, std::vector<double>> prev_param;
        // L-BFGS gradient vector.
        std::map<SPNNode *, std::vector<double>> lbfgs_grads;
        // Store the function values during the optimization.
        std::vector<double> train_funcs, valid_funcs;
        // Masks for inference.
        std::vector<bool> mask_false(num_var, false);
        std::vector<bool> mask_true(num_var, true);
        // Initialize SST based on the structure of the network.
        for (SPNNode *pt : spn.top_down_order()) {
            if (pt->type() == SPNNodeType::SUMNODE) {
                // Store current gradient vector and the optimal parameter vector.
                sst.insert({pt, std::vector<double>(pt->num_children())});
                opt.insert({pt, std::vector<double>(pt->num_children())});
                // Store previous gradient vector and previous parameter vector.
                prev_grads.insert({pt, std::vector<double>(pt->num_children(), 0.0)});
                prev_param.insert({pt, std::vector<double>(pt->num_children(), 0.0)});
                // Store gradient vectors and parameter vectors in the history window.
                for (int i = 0; i < history_window_; ++i) {
                    history_param[i].insert({pt, std::vector<double>(pt->num_children())});
                    history_grads[i].insert({pt, std::vector<double>(pt->num_children())});
                }
                // Store the L-BFGS gradient vector.
                lbfgs_grads.insert({pt, std::vector<double>(pt->num_children())});
            }
        }
        // Start projected gradient descent.
        if (verbose) {
            std::cout << "#iteration" << "," << "train-lld" << "," << "valid-lld" << std::endl;
        }
        int max_hist = 0, curr_hist = 0;
        for (size_t t = 0; t < num_iters_; ++t) {
            // Not good fitting, shrinking the weight.
            if (t > 1 && train_funcs[t-1] < train_funcs[t-2]) {
                lrate *= shrink_weight_;
            }
            // Clean previous records.
            train_logps = 0.0;
            valid_logps = 0.0;
            for (auto &pvec : sst) {
                for (size_t k = 0; k < pvec.second.size(); ++k) {
                    pvec.second[k] = 0.0;
                }
            }
            // Compute the gradient vector in the current iteration.
            for (size_t n = 0; n < num_trains + num_valids; ++n) {
                // Bottom-up and top-down passes of the network.
                if (n < num_trains) {
                    train_logps += spn.EvalDiff(trains[n], mask_false);
                    // Compute gradients.
                    for (SPNNode *pt : spn.top_down_order()) {
                        if (pt->type() == SPNNodeType::SUMNODE) {
                            // Projected gradient descent update formula.
                            for (size_t k = 0; k < pt->num_children(); ++k) {
                                sst[pt][k] += exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                            }
                        }
                    }
                } else {
                    valid_logps += spn.EvalDiff(valids[n - num_trains], mask_false);
                }
            }
            // Propagate value of all one vector.
            spn.EvalDiff(all_one, mask_true);
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        sst[pt][k] -= num_trains * exp(pt->dr() + pt->children()[k]->fr() - spn.root_->fr());
                        sst[pt][k] /= num_trains;
                    }
                }
            }
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
            // Update current history record.
            for (SPNNode *pt : spn.top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    for (size_t k = 0; k < pt->num_children(); ++k) {
                        // Update history and previous.
                        history_param[curr_hist][pt][k] = ((SumNode *) pt)->weights()[k] - prev_param[pt][k];
                        history_grads[curr_hist][pt][k] = sst[pt][k] - prev_grads[pt][k];
                        prev_param[pt][k] = ((SumNode *) pt)->weights()[k];
                        prev_grads[pt][k] = sst[pt][k];
                        // Update the L-BFGS vector.
                        lbfgs_grads[pt][k] = sst[pt][k];
                    }
                }
            }

            curr_hist %= history_window_;
            max_hist = std::min(history_window_, t);
            double alpha = 0.0, pho = 0.0;
            int hist_index = 0;
            for (int h = 0; h < max_hist; ++h) {
                hist_index = (curr_hist - h + history_window_) % history_window_;
                // Compute the inner product coefficient alpha_h in the L-BFGS algorithm.
                alpha = 0.0;
                pho = 0.0;
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            pho += history_param[hist_index][pt][k] * history_grads[hist_index][pt][k];
                            alpha += history_param[hist_index][pt][k] * lbfgs_grads[pt][k];
                        }
                    }
                }
                history_phos[hist_index] = 1.0 / pho;
                history_alphas[hist_index] = alpha / pho;
                // Forward updating of the L-BFGS vector.
                for (SPNNode *pt : spn.top_down_order()) {
                    if (pt->type() == SPNNodeType::SUMNODE) {
                        for (size_t k = 0; k < pt->num_children(); ++k) {
                            lbfgs_grads[pt][k] -= history_alphas[hist_index] * history_grads[hist_index][pt][k];
                        }
                    }
                }
                // r = H_{-m} * lbfgs_grad. In this implementation, H_{-m} is fixed to be the identity matrix.
                
            }
            curr_hist += 1;
            // Weight update using projected gradient descent.
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
            // Stop criterion.
            if (t > 0 && fabs(train_funcs[t] - train_funcs[t-1]) < stop_thred_) {
                break;
            }
        }
        // Restore the optimal model weights during the optization.
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
}
