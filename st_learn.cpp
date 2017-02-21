//
// Created by Han Zhao on 20/02/2017.
//

#include "src/SPNetwork.h"
#include "src/utils.h"
#include "src/StructureLearning.h"

#include <fstream>
#include <queue>
#include <random>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using SPN::SPNetwork;
using SPN::LearnOptSPN;


int main(int argc, char *argv[]) {
    // Positional program parameters.
    std::string model_filename, training_filename, valid_filename, test_filename, algo_name;
    // Hyperparameters for LearnOptSPN.
    uint seed = 42;
    double eps = 0.01;
    // Building command line parser
    po::options_description desc("Please specify the following options");
    desc.add_options()
            // Positional program parameters.
            ("train", po::value<std::string>(&training_filename), "file path of training data")
            ("valid", po::value<std::string>(&valid_filename), "file path of validation data")
            ("test", po::value<std::string>(&test_filename), "file path of test data")
            ("model", po::value<std::string>(&model_filename), "file path of SPN to be saved")
            ("algo", po::value<std::string>(&algo_name), "batch algorithm")
            // Hyperparameters for structure learning algorithms.
            ("seed", po::value<uint>(&seed), "random seed")
            ("eps", po::value<double>(&eps), "proportion of the uniform distribution during learning");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (!vm.count("train") || !vm.count("valid") || !vm.count("test") ||
        !vm.count("model")) {
        std::cout << desc << std::endl;
        return -1;
    }
    // Load training and test data sets
    std::vector<std::vector<double>> training_data = SPN::utils::load_data(training_filename);
    std::vector<std::vector<double>> valid_data = SPN::utils::load_data(valid_filename);
    std::vector<std::vector<double>> test_data = SPN::utils::load_data(test_filename);

    std::cout << "Model path: " << model_filename << std::endl;
    std::cout << "Number of instances in training set = " << training_data.size() << std::endl;
    std::cout << "Number of instances in validation set = " << valid_data.size() << std::endl;
    std::cout << "Number of instances in test set = " << test_data.size() << std::endl;

    size_t num_train = training_data.size(), num_valid = valid_data.size(), num_test = test_data.size();
    if (training_data[0].size() != valid_data[0].size()
        || valid_data[0].size() != test_data[0].size()) {
        std::cerr << "Trainging data, validation data and test data are not consistent in dimension" << std::endl;
        return -1;
    }
    // Construct SPN.
    LearnOptSPN st_learner;
    SPNetwork *spn = st_learner.learn(training_data, valid_data, eps, true);
    spn->init();
    std::cout << "Uniform hyperparameter eps = " << eps << std::endl;
    std::cout << "Network statistics after initialization: " << std::endl;
    std::cout << "Network height: " << spn->height() << std::endl;
    std::cout << "Network size: " << spn->size() << std::endl;
    std::cout << "Network number of nodes: " << spn->num_nodes() << std::endl;
    std::cout << "Network number of edges: " << spn->num_edges() << std::endl;
    std::cout << "Network number of varnodes: " << spn->num_var_nodes() << std::endl;
    std::cout << "Network number of sumnodes: " << spn->num_sum_nodes() << std::endl;
    std::cout << "Network number of prodnodes: " << spn->num_prod_nodes() << std::endl;
    std::cout << "**********************************" << std::endl;
    // Compute test set average log-likelihoods
    std::clock_t t_start = std::clock();
    vector<double> test_logps = spn->logprob(test_data);
    std::clock_t t_end = std::clock();
    std::cout << "CPU time = " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC << " milliseconds\n";

    double avg_logp = 0.0;
    for (double ll : test_logps) avg_logp += ll;
    avg_logp /= num_test;
    std::cout << "Average test log-likelihoods = " << avg_logp << std::endl;

    vector<double> train_logps = spn->logprob(training_data);
    avg_logp = 0.0;
    for (double ll : train_logps) avg_logp += ll;
    avg_logp /= num_train;
    std::cout << "Average training log-likelihoods = " << avg_logp << std::endl;

    vector<double> valid_logps = spn->logprob(valid_data);
    avg_logp = 0.0;
    for (double ll : valid_logps) avg_logp += ll;
    avg_logp /= num_valid;
    std::cout << "Average validation log-likelihood = " << avg_logp << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    if (vm.count("model")) {
        SPN::utils::save(spn, model_filename);
    }
    delete spn;
    return 0;
}
