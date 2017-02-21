//
// Created by Han Zhao on 11/19/15.
//
#include "src/SPNNode.h"
#include "src/SPNetwork.h"
#include "src/utils.h"
#include "src/StreamParamLearning.h"

#include <fstream>
#include <random>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using SPN::SPNetwork;
using SPN::StreamParamLearning;
using SPN::StreamProjectedGD;
using SPN::StreamExpoGD;
using SPN::StreamSMA;
using SPN::StreamExpectMax;
using SPN::utils::split;

// The main difference between the online version of these algorithms and the
// streaming version of these algorithms lie in the fact that in the online
// setting we're allowed to revisit each training instance multiple times while
// in the streaming setting each instance should be visited only once.
// Furthermore, in the streaming setting normally the data set is so large that
// we cannot load them all at once into main memory, so instead we load each instance
// at one time.
int main(int argc, char *argv[]) {
    // Positional program parameters.
    std::string model_filename, training_filename, valid_filename, test_filename, stream_algo;
    std::string output_model_filename;
    // Hyperparameter for the streaming expectation maximization algorithm.
    uint seed = 42;
    double proj_eps = 1e-2;
    double lrate = 1e-1;
    // Building command line parser
    po::options_description desc("Please specify the following options");
    desc.add_options()
            // Positional program parameters.
            ("train", po::value<std::string>(&training_filename), "file path of training data")
            ("test", po::value<std::string>(&test_filename), "file path of test data")
            ("model", po::value<std::string>(&model_filename), "file path of SPN")
            ("output_model", po::value<std::string>(&output_model_filename), "file path of SPN to save")
            ("algo", po::value<std::string>(&stream_algo), "streaming algorithm")
            // Hyperparameters for EM.
            ("seed", po::value<uint>(&seed), "random seed")
            ("proj_eps", po::value<double>(&proj_eps), "projection constant")
            ("lrate", po::value<double>(&lrate), "learning rate");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (!vm.count("train") || !vm.count("test") || !vm.count("model") || !vm.count("algo")) {
        std::cout << desc << std::endl;
        return -1;
    }
    std::cout << "Loaded model: " << model_filename << std::endl;
    // Load and simplify the SPN.
    SPNetwork *spn = SPN::utils::load(model_filename);
    spn->init();
    std::cout << "Network statistics after initialization: " << std::endl;
    std::cout << "Network height: " << spn->height() << std::endl;
    std::cout << "Network size: " << spn->size() << std::endl;
    std::cout << "Network number of nodes: " << spn->num_nodes() << std::endl;
    std::cout << "Network number of edges: " << spn->num_edges() << std::endl;
    std::cout << "Network number of varnodes: " << spn->num_var_nodes() << std::endl;
    std::cout << "Network number of sumnodes: " << spn->num_sum_nodes() << std::endl;
    std::cout << "Network number of prodnodes: " << spn->num_prod_nodes() << std::endl;
    std::cout << "**********************************" << std::endl;
    // Streaming projected gradient descent.
    // Random initialization of model parameters.
    spn->set_random_params(seed);
    const auto &tokens = split(model_filename, '.');
    std::string data_name = split(tokens[0], '/')[1];
    StreamParamLearning *slearning = nullptr;
    // Select the concrete algorithm to be used.
    if (stream_algo == "pgd") {
        slearning = new StreamProjectedGD(proj_eps, lrate);
    } else if (stream_algo == "eg") {
        slearning = new StreamExpoGD(lrate);
    } else if (stream_algo == "sma") {
        slearning = new StreamSMA(lrate);
    } else if (stream_algo == "em" || stream_algo == "cccp") {
        slearning = new StreamExpectMax(lrate);
    } else {
        std::cerr << "Please choose from pgd, eg, sma or em" << std::endl;
        std::exit(-1);
    }
    std::cout << slearning->algo_name() << std::endl;
    std::cout << "Optimization hyperparameters:" << std::endl;
    std::cout << "Random seed = " << seed << std::endl;
    std::cout << "Projection constant = " << proj_eps << std::endl;
    std::cout << "Learning rate = " << lrate << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::ifstream fin(training_filename, std::ifstream::in);
    if (!fin) {
        std::cerr << "Failed to open file: " << training_filename << std::endl;
        std::exit(-1);
    }
    std::string line;
    std::vector<double> input_inst;
    std::vector<std::string> terms;
    size_t num_train = 0, num_test = 0;
    std::clock_t t_start = std::clock();
    // In case there is interruption during the training.
    while (std::getline(fin, line)) {
        terms = split(line, ',');
        input_inst.clear();
        for (const std::string &term : terms)
            input_inst.push_back(std::stod(term));
        slearning->fit(input_inst, *spn);
        num_train += 1;
    }
    fin.close();
    std::clock_t t_end = std::clock();
    spn->weight_projection(1e-3);
    std::cout << "Time used for training = " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
              << " milliseconds" << std::endl;
    std::cout << "Number of streaming instances = " << num_train << std::endl;
    // Testing.
    fin.open(test_filename, std::ifstream::in);
    if (!fin) {
        std::cerr << "Failed to open file: " << test_filename << std::endl;
        std::exit(-1);
    }
    t_start = std::clock();
    double tlogp = 0.0;
    while (std::getline(fin, line)) {
        terms = split(line, ',');
        input_inst.clear();
        for (const std::string &term : terms)
            input_inst.push_back(std::stoi(term));
        tlogp += spn->logprob(input_inst);
        num_test += 1;
    }
    tlogp /= num_test;
    t_end = std::clock();
    std::cout << "Time used for testing = " << 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC
              << " milliseconds" << std::endl;
    std::cout << "Number of test instances = " << num_test << std::endl;
    std::cout << "Average log-likelihoods = " << tlogp << std::endl;
    if (vm.count("output_model")) {
        SPN::utils::save(spn, output_model_filename);
    }
    delete spn;
    delete slearning;
    return 0;
}


