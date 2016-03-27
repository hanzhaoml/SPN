//
// Created by Han Zhao on 11/19/15.
//

#include "src/SPNNode.h"
#include "src/SPNetwork.h"
#include "src/utils.h"
#include "src/BatchParamLearning.h"

#include <fstream>
#include <queue>
#include <random>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using namespace std;
using SPN::SPNNode;
using SPN::SumNode;
using SPN::ProdNode;
using SPN::VarNode;
using SPN::SPNNodeType;
using SPN::SPNetwork;
using SPN::BatchParamLearning;
using SPN::ProjectedGD;
using SPN::ExpoGD;
using SPN::SMA;
using SPN::ExpectMax;
using SPN::CollapsedVB;
using SPN::LBFGS;
using SPN::utils::split;

int main(int argc, char *argv[]) {
    // Positional program parameters.
    std::string model_filename, training_filename, valid_filename, test_filename, algo_name;
    std::string output_model_filename;
    // Hyperparameters for projected gradient descent algorithm.
    uint seed = 42;
    int num_iters = 50;
    uint history_window = 20;
    double stop_thred = 1e-2;
    double lap_smooth = 1e-3;
    double proj_eps = 1e-2;
    double lrate = 1e-1;
    double shrink_weight = 8e-1;
    double prior_scale = 100.0;
    double train_fraction = 1.0;
    // Building command line parser
    po::options_description desc("Please specify the following options");
    desc.add_options()
            // Positional program parameters.
            ("train", po::value<std::string>(&training_filename), "file path of training data")
            ("valid", po::value<std::string>(&valid_filename), "file path of validation data")
            ("test", po::value<std::string>(&test_filename), "file path of test data")
            ("model", po::value<std::string>(&model_filename), "file path of SPN")
            ("output_model", po::value<std::string>(&output_model_filename), "file path of SPN to save")
            ("algo", po::value<std::string>(&algo_name), "batch algorithm")
            ("train_fraction", po::value<double>(&train_fraction), "fraction of training data")
            // Hyperparameters for training algorithms.
            ("seed", po::value<uint>(&seed), "random seed")
            ("num_iters", po::value<int>(&num_iters), "maximum number of iterations")
            ("stop_thred", po::value<double>(&stop_thred), "stop criterion for consecutive function values")
            ("proj_eps", po::value<double>(&proj_eps), "projection constant for ProjectedGD algorithm")
            ("shrink_weight", po::value<double>(&shrink_weight), "shrinking weight during line search")
            ("lrate", po::value<double>(&lrate), "learning rate")
            ("lap_smooth", po::value<double>(&lap_smooth), "smoothing parameter")
            ("prior_scale", po::value<double>(&prior_scale), "scale parameter the prior distritbuion");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (!vm.count("train") || !vm.count("valid") || !vm.count("test") ||
        !vm.count("model") || !vm.count("algo")) {
        std::cout << desc << std::endl;
        return -1;
    }
    // Load training and test data sets
    std::vector<std::vector<double>> training_data = SPN::utils::load_data(training_filename);
    std::vector<std::vector<double>> valid_data = SPN::utils::load_data(valid_filename);
    std::vector<std::vector<double>> test_data = SPN::utils::load_data(test_filename);

    std::cout << "Loaded model: " << model_filename << std::endl;
    std::cout << "Number of instances in training set = " << training_data.size() << std::endl;
    std::cout << "Number of instances in validation set = " << valid_data.size() << std::endl;
    std::cout << "Number of instances in test set = " << test_data.size() << std::endl;

    size_t num_train = training_data.size(), num_valid = valid_data.size(), num_test = test_data.size();
    if (training_data[0].size() != valid_data[0].size()
        || valid_data[0].size() != test_data[0].size()) {
        std::cerr << "Trainging data, validation data and test data are not consistent in dimension" << std::endl;
        return -1;
    }
    // Load and simplify SPN
    SPNetwork *spn = SPN::utils::load(model_filename);
    spn->init();
    std::cout << "Network statistics after initialization: " << std::endl;
    cout << "Network height: " << spn->height() << endl;
    cout << "Network size: " << spn->size() << endl;
    cout << "Network number of nodes: " << spn->num_nodes() << endl;
    cout << "Network number of edges: " << spn->num_edges() << endl;
    cout << "Network number of varnodes: " << spn->num_var_nodes() << endl;
    cout << "Network number of sumnodes: " << spn->num_sum_nodes() << endl;
    cout << "Network number of prodnodes: " << spn->num_prod_nodes() << endl;
    cout << "**********************************" << endl;
    BatchParamLearning *learning = nullptr;
    if (algo_name == "pgd") {
        learning = new ProjectedGD(num_iters, proj_eps, stop_thred, lrate, shrink_weight);
    } else if (algo_name == "eg") {
        learning = new ExpoGD(num_iters, stop_thred, lrate, shrink_weight);
    } else if (algo_name == "sma") {
        learning = new SMA(num_iters, stop_thred, lrate, shrink_weight);
    } else if (algo_name == "em") {
        learning = new ExpectMax(num_iters, stop_thred, lap_smooth);
    } else if (algo_name == "cvb") {
        learning = new CollapsedVB(num_iters, stop_thred, lrate, prior_scale, seed);
    } else if (algo_name == "lbfgs") {
        learning = new LBFGS(num_iters, proj_eps, stop_thred, lrate, shrink_weight, history_window);
    } else {
        std::cerr << "Please choose from pgd, eg, sma, em or cvb" << std::endl;
        std::exit(-1);
    }
    // Compute test set average log-likelihoods
    std::clock_t t_start = std::clock();
    vector<double> logps = spn->logprob(test_data);
    std::clock_t t_end = std::clock();
    std::cout << "CPU time = " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " milliseconds\n";
    double avg_logp = 0.0;
    for (double ll : logps) avg_logp += ll;
    avg_logp /= num_test;
    std::cout << "Average log-likelihoods = " << avg_logp << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << learning->algo_name() << std::endl;
    std::cout << "Optimization hyperparameters:" << std::endl;
    std::cout << "Random seed = " << seed << std::endl;
    std::cout << "Number of maximum iterations = " << num_iters << std::endl;
    std::cout << "Stopping criterion = " << stop_thred << std::endl;
    std::cout << "Laplacian smoothing = " << lap_smooth << std::endl;
    std::cout << "Projection constant = " << proj_eps << std::endl;
    std::cout << "Shrinking weight = " << shrink_weight << std::endl;
    std::cout << "Learning rate = " << lrate << std::endl;
    std::cout << "Scale parameter for prior = " << prior_scale << std::endl;
    std::cout << "Training fraction of the data = " << train_fraction << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    // Random initialization of model parameters.
    spn->set_random_params(seed);
    const auto &tokens = split(model_filename, '.');
    string data_name = split(tokens[0], '/')[1];
    t_start = std::clock();
    // Use only a fraction of the data set.
    uint real_num_train = num_train * train_fraction;
    training_data.erase(training_data.begin()+real_num_train, training_data.end());
    learning->fit(training_data, valid_data, *spn, true);
    t_end = std::clock();
    std::cout << "Training time = " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
              << " milliseconds" << std::endl;
    logps = spn->logprob(test_data);
    avg_logp = 0.0;
    for (double ll : logps) avg_logp += ll;
    avg_logp /= num_test;
    std::cout << "Test average log-likelihoods = " << avg_logp << std::endl;
    std::cout << "**********************************************************" << std::endl;
    if (vm.count("output_model")) {
        SPN::utils::save(spn, output_model_filename);
    }
    delete spn;
    delete learning;
    return 0;
}
