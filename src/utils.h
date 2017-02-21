#include "SPNetwork.h"

#include <string>
// Declarations of utility functions

namespace SPN {
    namespace utils {
        // Helper functions for split string
        std::vector<std::string> split(const std::string &, char);

        // Load an SPN from file.
        SPNetwork *load(std::string);

        // Save an SPN to file.
        void save(SPNetwork*, std::string);

        std::vector<std::vector<double>> load_data(std::string);
    }
}