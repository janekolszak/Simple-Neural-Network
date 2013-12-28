#ifndef SNN_TEST_UTILS_HPP
#define SNN_TEST_UTILS_HPP

#include <boost/test/unit_test.hpp>
#include <snn/types.hpp>

using namespace snn;

template <typename NeuronType, typename... LearningParams>
void testIfLearned(BasicPerceptron<NeuronType, LearningParams... > &perceptron,
                   SnnDataset &trainSet, const double accurancy) {
    for (auto itSet =  trainSet.begin();
            itSet != trainSet.end();
            std::advance (itSet, 2)) {
        SnnValVec &in = *itSet;
        SnnValVec &out = *std::next(itSet);

        perceptron.forward(in);
        auto perOutput = perceptron.getOutputs();
        auto itOutputs = perOutput.begin();
        auto itTrain = out.begin();
        for (; itOutputs != perOutput.end();
                ++itOutputs, ++itTrain) {
            BOOST_REQUIRE(abs(*itOutputs - *itTrain) < accurancy);
        }
    }
}

#endif