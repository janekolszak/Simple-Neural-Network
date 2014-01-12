#ifndef SNN_SCALAR_LEARNING_RATE_WITH_MOMENTUM_HPP
#define SNN_SCALAR_LEARNING_RATE_WITH_MOMENTUM_HPP

#include <vector>

#include <snn/types.hpp>
#include <snn/basic/networks/basic_perceptron.hpp>
#include <snn/basic/neurons/basic_neuron.hpp>

namespace snn {

struct ScalarLearningRateMomentumNeuron: BasicNeuron<SnnVal, SnnVal> {

    SnnValVec _previousWeightChange;
    SnnVal _prevBiasChange = 0.0;

    ScalarLearningRateMomentumNeuron(): BasicNeuron<SnnVal, SnnVal>() {}

    ScalarLearningRateMomentumNeuron(SnnVal value,
                                     ActivationFunction *activation)
        : BasicNeuron<SnnVal, SnnVal>(value, activation) {}

    ScalarLearningRateMomentumNeuron &operator=(SnnVal newValue) {
        _value = newValue;
        return *this;
    }

    void connectSource(BasicNeuron<SnnVal, SnnVal> &source, const SnnVal &weight)
    {
        BasicNeuron<SnnVal, SnnVal>::connectSource(source, weight);
        _previousWeightChange.push_back(0.0);
    }

    void learn(SnnVal learningRate, SnnVal momentum)
    {
        auto itW = _inputWeights.begin();
        auto itV = _inputValues.begin();
        auto itP = _previousWeightChange.begin();
        for (; itW != _inputWeights.end();
                ++itW, ++itV, ++itP) {
            *itP = *itV * learningRate * _delta + momentum **itP;
            *itW += *itP;
        }

        _prevBiasChange = learningRate * _delta + momentum * _prevBiasChange;
        _bias += _prevBiasChange;

    }
};

struct ScalarLearningRateMomentumPerceptron: BasicPerceptron<ScalarLearningRateMomentumNeuron, SnnVal, SnnVal> {

    ScalarLearningRateMomentumPerceptron(std::initializer_list<size_t> layerSizes)
        : BasicPerceptron<ScalarLearningRateMomentumNeuron, SnnVal, SnnVal>(layerSizes) {}

    void learn(SnnVal learningRate, SnnVal momentum) {
        for (auto &layer : _layers)
            layer.learn(learningRate, momentum);
    }
};


void train(ScalarLearningRateMomentumPerceptron &perceptron,
           SnnDataset dataset,
           size_t numEpochs,
           SnnVal learningRate = 0.3,
           SnnVal momentum = 0.9) {

    for (size_t i = 0; i < numEpochs; ++i) {
        for (auto itSet =  dataset.begin();
                itSet != dataset.end();
                std::advance(itSet, 2)) {
            SnnValVec &in = *itSet;
            SnnValVec &out = *std::next(itSet);
            // cout << "-----------------------------------" << endl;
            // cout << "WAGI " <<   perceptron.getWeights() << endl;

            perceptron.forward(in);
            // cout << "OUT " << perceptron.getOutputs() << endl;
            perceptron.backward(out);

            perceptron.learn(learningRate, momentum);
        }
    }
}

} // namespace snn

#endif