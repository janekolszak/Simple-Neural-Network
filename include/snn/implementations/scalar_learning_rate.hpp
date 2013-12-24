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
        auto itW = _inputWeights.begin();
        auto itV = _inputValues.begin();
        for (; itW != _inputWeights.end();
                ++itW, ++itV) {
            *itW += *itV * learningRate * _delta;
        }
        _bias += learningRate * _delta;
        // cout << _bias << endl;
    }
};

struct ScalarLearningRatePerceptron: BasicPerceptron<ScalarLearningRateNeuron, SnnVal> {

    ScalarLearningRatePerceptron(std::initializer_list<size_t> layerSizes)
        : BasicPerceptron<ScalarLearningRateNeuron, SnnVal>(layerSizes) {}

    void learn(SnnVal learningRate) {
        for (auto &layer : _layers)
            layer.learn(learningRate);
    }
};


void train(ScalarLearningRatePerceptron &perceptron, SnnDataset dataset, size_t numEpochs) {
    SnnVal learningRate = 0.7;

    for (size_t i = 0; i < numEpochs; ++i) {
        for (auto itSet =  dataset.begin();
                itSet != dataset.end();
                std::advance (itSet, 2)) {
            SnnValVec &in = *itSet;
            SnnValVec &out = *std::next(itSet);
            // cout << "-----------------------------------" << endl;
            // cout << "WAGI " <<   perceptron.getWeights() << endl;

            perceptron.forward(in);
            // cout << "OUT " << perceptron.getOutputs() << endl;
            perceptron.backward(out);

            perceptron.learn(learningRate);
        }
    }
}

} // namespace snn

#endif