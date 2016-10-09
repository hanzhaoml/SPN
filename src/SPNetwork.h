//
// Created by Han Zhao on 5/7/15.
//

#ifndef SPN_SL_SPNETWORK_H
#define SPN_SL_SPNETWORK_H

#include "SPNNode.h"
#include "fmath.hpp"

#include <unordered_map>
#include <unordered_set>
#include <cmath>

using fmath::log;
using fmath::exp;

namespace SPN {
    class SPNetwork {
    public:
        SPNetwork(SPNNode *);

        virtual ~SPNetwork();

        // Getters
        int size() const {
            return size_;
        }

        int height() const {
            return height_;
        }

        int num_nodes() const {
            return num_nodes_;
        }

        int num_edges() const {
            return num_edges_;
        }

        int num_var_nodes() const {
            return num_var_nodes_;
        }

        int num_sum_nodes() const {
            return num_sum_nodes_;
        }

        int num_prod_nodes() const {
            return num_prod_nodes_;
        }

        // BFS implementation for the inference of SPNs
        double inference(const std::vector<double> &, bool verbose = false);

        std::vector<double> inference(const std::vector<std::vector<double>> &, bool verbose = false);

        double logprob(const std::vector<double> &, bool verbose = false);

        std::vector<double> logprob(const std::vector<std::vector<double>> &, bool verbose = false);

        const std::vector<SPNNode *> &bottom_up_order() const {
            return forward_order_;
        }

        const std::vector<SPNNode *> &top_down_order() const {
            return backward_order_;
        }

        const std::vector<VarNode *> dist_nodes() const {
            return dist_nodes_;
        }

        // Set the fr and dr values at each node with input x and return the
        // log-probability of the input vector, mask is used to indicate whether
        // the corresponding feature should be integrated/marginalized out or not.
        // If mask[i] = true, then the ith feature will be integrated out, otherwise
        // not.
        double EvalDiff(const std::vector<double> &input, const std::vector<bool> &mask);

        // Initialize the SPN, do the following tasks:
        // 1, Remove connected sum nodes and product nodes
        // 2, Compute statistics about the network topology
        // 3, Build the bottom-up and top-down visiting order of nodes in SPN
        void init();

        // Drop the existing model parameters and initialize using random seed.
        void set_random_params(uint seed);

        // Project each nonlocally normalized SPN into an SPN with locally
        // normalized weights.
        void weight_projection(double smooth = 0.0);

        // Output the network
        void print(std::ostream &);

    private:
        // Delegate function call
        template<typename Callable>
        void bfs_traverse(Callable &&);

        void condense_(SPNNode *, std::unordered_set<SPNNode *> &);

        void build_order_();

        void compute_statistics_();

        bool check_structure_();

        // Root handler of SPN
        SPNNode *root_ = nullptr;
        // Id -> Node mapping
        std::unordered_map<int, SPNNode *> id2node_;
        // Bottom-up order of all the nodes in the network, including indicator nodes.
        std::vector<SPNNode *> forward_order_;
        // Top-down order of all the nodes in the network, including indicator nodes.
        std::vector<SPNNode *> backward_order_;
        // Store all the leaf distribution nodes for fast processing.
        std::vector<VarNode *> dist_nodes_;

        // Network topology
        int size_ = 0;
        int height_ = 0;

        int num_nodes_ = 0;
        int num_edges_ = 0;
        int num_var_nodes_ = 0;
        int num_sum_nodes_ = 0;
        int num_prod_nodes_ = 0;

        // Friend declaration for Algorithmic classes
        friend class ExpectMax;

        friend class CollapsedVB;

        friend class SMA;

        friend class ExpoGD;

        friend class ProjectedGD;

        friend class LBFGS;

        friend class OnlineExpectMax;

        friend class OnlineCollapsedVB;

        friend class OnlineADF;

        friend class OnlineBMM;

        friend class OnlineExpoGD;

        friend class OnlineProjectedGD;

        friend class OnlineSMA;

        friend class StreamProjectedGD;

        friend class StreamExpoGD;

        friend class StreamSMA;

        friend class StreamExpectMax;
    };
}


#endif //SPN_SL_SPNETWORK_H
