//
// Created by Han Zhao on 20/02/2017.
//

#ifndef SPN_EM_STRUCTURELEARNING_H
#define SPN_EM_STRUCTURELEARNING_H

#include "SPNetwork.h"

#include <vector>

namespace SPN {
    // Learning the struture of the optimal discrete SPNs.
    class LearnOptSPN {
    public:
        LearnOptSPN() = default;

        virtual ~LearnOptSPN() = default;

        SPNetwork *learn(const std::vector<std::vector<double>> &trains,
                         const std::vector<std::vector<double>> &valids, double eps, bool verbose=false);
    };
}

#endif //SPN_EM_STRUCTURELEARNING_H
