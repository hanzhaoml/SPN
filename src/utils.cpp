//
// Created by Han Zhao on 5/6/15.
//
#include "SPNNode.h"
#include "SPNetwork.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <queue>
#include <stack>

namespace SPN {
    namespace utils {

        std::vector<std::string> split(const std::string &str, char delim) {
            std::vector<std::string> tokens;
            std::stringstream ss(str);
            std::string token;
            while (std::getline(ss, token, delim)) {
                tokens.push_back(token);
            }
            return tokens;
        }

        void save(SPNetwork *spn, std::string filename) {
            std::ofstream fout(filename);
            if (!fout) {
                std::cerr << "Open filename: " << filename << " failed" << std::endl;
                std::exit(-1);
            }
            fout << "##NODES##\n";
            for (SPNNode *pt : spn->top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE) {
                    if (pt->scope().size() == 1) {
                        // Leaf sum nodes.
                        fout << pt->id() << ",LEAVE," << pt->scope()[0] << ",";
                        fout << ((SumNode *)pt)->weights()[0] << ","
                             << ((SumNode *)pt)->weights()[1] << "\n";
                    } else {
                        // Internal sum nodes.
                        fout << pt->id() << ",SUM\n";
                    }
                } else if (pt->type() == SPNNodeType::PRODNODE) {
                    fout << pt->id() << ",PRD\n";
                }
            }
            fout << "##EDGES##\n";
            for (SPNNode *pt : spn->top_down_order()) {
                if (pt->type() == SPNNodeType::SUMNODE && pt->scope().size() > 1) {
                    for (size_t i = 0; i < pt->num_children(); ++i) {
                        fout << pt->id() << "," << pt->children()[i]->id() << ","
                             << ((SumNode *)pt)->weights()[i] << "\n";
                    }
                } else if (pt->type() == SPNNodeType::PRODNODE) {
                    for (size_t i = 0; i < pt->num_children(); ++i) {
                        fout << pt->id() << "," << pt->children()[i]->id() << "\n";
                    }
                }
            }
            fout.close();
        }

        SPNetwork *load(std::string filename) {
            // scope and nodes
            std::unordered_map<int, SPNNode *> id2node;
            // Use a specific map to store indicator variables,
            // where X2=false -> 2*2 = 4 and X2 = true -> 2*2 + 1
            std::unordered_map<int, std::pair<SPNNode *, SPNNode *>> varnodes;
            std::ifstream fin(filename, std::ifstream::in);
            std::string line;
            size_t num_nodes = 0, num_edges = 0;
            int var_index;
            // Header ##NODES##
            int id1, id2;
            // Build graph first
            if (!fin) {
                std::cerr << "Open filename: " << filename << " failed" << std::endl;
                std::exit(-1);
            }
            while (std::getline(fin, line)) {
                auto tokens = split(line, ',');
                // ##NODES## or ##EDGES##
                if (tokens.size() == 1) continue;
                id1 = std::stoi(tokens[0]);
                if (tokens[1] == "SUM") {
                    // Sum node
                    SPNNode *node = new SumNode(id1);
                    assert(id2node.find(id1) == id2node.end());
                    id2node.insert({id1, node});
                    num_nodes += 1;
                } else if (tokens[1] == "PRD") {
                    // Product node
                    SPNNode *node = new ProdNode(id1);
                    assert(id2node.find(id1) == id2node.end());
                    id2node.insert({id1, node});
                    num_nodes += 1;
                } else if (tokens.size() == 2) {
                    // Edge from a product node.
                    id2 = std::stoi(tokens[1]);
                    SPNNode *parent = id2node[id1], *child = id2node[id2];
                    parent->add_child(child);
                    child->add_parent(parent);
                    num_edges += 1;
                } else if (tokens.size() == 3) {
                    // Edge from a sum node
                    id2 = std::stoi(tokens[1]);
                    SPNNode *parent = id2node[id1], *child = id2node[id2];
                    parent->add_child(child);
                    child->add_parent(parent);
                    ((SumNode *) parent)->add_weight(std::stod(tokens[2]));
                    num_edges += 1;
                } else {
                    // tokens.size() > 3, univariate distribution node.
                    std::transform(tokens[1].begin(), tokens[1].end(), tokens[1].begin(), ::toupper);
                    var_index = std::stoi(tokens[2]);
                    // Binary distribution, the first value corresponds to Pr(X=false),
                    // the second value corresponds to Pr(X=true).
                    if (tokens[1].compare("BINNODE") == 0) {
                        SPNNode *node = new SumNode(id1, std::vector<int>{var_index},
                                                    std::vector<double>{std::stod(tokens[3]), std::stod(tokens[4])});
                        num_nodes += 1;
                        num_edges += 2;
                        // Process all the scopes later...
                        node->clear_scope();
                        assert(id2node.find(id1) == id2node.end());
                        assert(node->scope().size() == 0);
                        id2node.insert({id1, node});
                        if (varnodes.find(var_index) == varnodes.end()) {
                            SPNNode *varnode_f = new BinNode(-1, var_index, 0.0);
                            SPNNode *varnode_t = new BinNode(-1, var_index, 1.0);
                            varnodes.insert({var_index, std::make_pair(varnode_f, varnode_t)});
                            // Build double connections
                            node->add_child(varnode_f);
                            node->add_child(varnode_t);
                            varnode_f->add_parent(node);
                            varnode_t->add_parent(node);
                            num_nodes += 2;
                        } else {
                            auto pair = varnodes[var_index];
                            // Build double connections
                            node->add_child(pair.first);
                            node->add_child(pair.second);
                            pair.first->add_parent(node);
                            pair.second->add_parent(node);
                        }
                    } else if (tokens[1].compare("NORMALNODE") == 0) {
                        // Normal distribution, the first value corresponds to the mean,
                        // the second value corresponds to the variance.
                        SPNNode *varnode = new NormalNode(id1, var_index,
                                                          std::stod(tokens[3]), std::stod(tokens[4]));
                        assert(id2node.find(id1) == id2node.end());
                        id2node.insert({id1, varnode});
                        num_nodes += 1;
                    } else {
                        std::cerr << "Error, leaf univariate distribution " << tokens[1]
                            << " not supported!" << std::endl;
                        std::exit(-1);
                    }
                }
            }
            fin.close();
            std::vector<SPNNode *> roots;
            for (const auto &kv : id2node) {
                if (kv.second->num_parents() == 0)
                    roots.push_back(kv.second);
            }
            // There must be only one root node
            assert(roots.size() == 1);
            SPNNode *root = roots[0];
            SPNetwork *spn = new SPNetwork(root);
            return spn;
        }

        std::vector<std::vector<double>> load_data(std::string filename) {
            std::ifstream fin(filename, std::ifstream::in);
            if (!fin) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                std::exit(-1);
            }
            std::string line;
            std::vector<std::vector<double>> data;
            while (std::getline(fin, line)) {
                std::vector<std::string> tokens = split(line, ',');
                std::vector<double> instance;
                for (const std::string &token : tokens)
                    instance.push_back(std::stod(token));
                data.push_back(instance);
            }
            fin.close();
            return data;
        }
    }
}