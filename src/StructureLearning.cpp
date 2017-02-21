//
// Created by Han Zhao on 20/02/2017.
//
#include "StructureLearning.h"

#include <map>

namespace SPN {
    SPNetwork *LearnOptSPN::learn(const std::vector<std::vector<double>> &trains,
                                  const std::vector<std::vector<double>> &valids, double eps, bool verbose) {
        std::map<std::vector<int>, size_t> freq;
        size_t num_insts = trains.size();
        size_t num_feats = trains[0].size();
        for (const auto &inst : trains) {
            freq[std::vector<int>(inst.begin(), inst.end())] += 1;
        }
        if (verbose) {
            std::cout << "In LearnOptSPN: " << std::endl;
            std::cout << "Number of unique instances in the training set = " << freq.size() << std::endl;
        }
        int id = 0;
        SumNode *root = new SumNode(id);
        // Use a specific map to store indicator variables,
        // where X2=false -> 2*2 = 4 and X2 = true -> 2*2 + 1
        std::vector<std::pair<SPNNode *, SPNNode *>> varnodes;
        for (size_t i = 0; i < num_feats; ++i)  {
            SPNNode *varnode_f = new BinNode(++id, i, 0.0);
            SPNNode *varnode_t = new BinNode(++id, i, 1.0);
            varnodes.push_back(std::make_pair(varnode_f, varnode_t));
        }
        // Create product node for data distribution.
        for (const auto &pair : freq) {
            SPNNode *node = new ProdNode(++id);
            node->add_parent(root);
            const std::vector<int> &inst = pair.first;
            for (size_t i = 0; i < inst.size(); ++i) {
                if (!inst[i]) {
                    node->add_child(varnodes[i].first);
                    varnodes[i].first->add_parent(node);
                } else {
                    node->add_child(varnodes[i].second);
                    varnodes[i].second->add_parent(node);
                }
            }
            root->add_child(node);
            root->add_weight((1.0 - eps) * pair.second / num_insts);
        }
        // Create sum node for uniform distribution.
        ProdNode *unif = new ProdNode(++id);
        root->add_child(unif);
        root->add_weight(eps);
        unif->add_parent(root);
        for (size_t i = 0; i < num_feats; ++i) {
            SumNode *node = new SumNode(++id);
            unif->add_child(node);
            node->add_parent(unif);
            node->add_child(varnodes[i].first);
            node->add_weight(0.5);
            node->add_child(varnodes[i].second);
            node->add_weight(0.5);
            varnodes[i].first->add_parent(node);
            varnodes[i].second->add_parent(node);
        }
        SPNetwork *spn = new SPNetwork(root);
        return spn;
    }
}
