#ifndef SNN_SCALAR_LEARNING_RATE_HPP
#define SNN_SCALAR_LEARNING_RATE_HPP

#include <snn/types.hpp>
#include <snn/basic/networks/basic_perceptron.hpp>
#include <snn/basic/neurons/basic_neuron.hpp>

namespace snn {

struct ScalarLearningRateNeuron: BasicNeuron<SnnVal> {

    ScalarLearningRateNeuron(): BasicNeuron<SnnVal>() {}
    ScalarLearningRateNeuron(SnnVal value,
                             SnnVal (*activation)(SnnVal),
                             SnnVal (*activationDerivative)(SnnVal))
        : BasicNeuron<SnnVal>(value,
                              activation,
                              activationDerivative) {}

    ScalarLearningRateNeuron &operator=(SnnVal newValue) {
        _value = newValue;
        return *this;
    }

    void learn(SnnVal learningRate)
    {
        SnnVal valueDerivative = _activationDerivative(_inputSum);
        for (auto &weight : _inputWeights) {
            weight += learningRate * _delta *  valueDerivative;
        }
        _bias += learningRate * _delta *  valueDerivative;
    }
};

// struct ScalarLearningRatePerceptron: BasicPerceptron<ScalarLearningRateNeuron, SnnVal> {

//     ScalarLearningRatePerceptron(std::initializer_list<size_t> layerSizes)
//         : BasicPerceptron<ScalarLearningRateNeuron, SnnVal>(layerSizes) {}

//     void learn(SnnVal learningRate) {
//         for (auto &layer : _layers)
//             layer.learn(learningRate);
//     }
// };

} // namespace snn

#endif