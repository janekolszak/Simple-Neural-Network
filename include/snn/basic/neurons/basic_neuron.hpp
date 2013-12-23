#ifndef SNN_BASIC_NEURON_HPP
#define SNN_BASIC_NEURON_HPP

#include <snn/types.hpp>

namespace snn {

template <typename... LearningParams>
struct BasicNeuron {

    SnnVal _value    = 0.0;
    SnnVal _inputSum = 0.0; ///< linear sum of inputs
    SnnVal _bias     = 0.0;
    SnnVal _delta    = 0.0;

    SnnValVec    _inputWeights;
    SnnValRefVec _outputWeights;
    SnnValRefVec _inputValues;
    SnnValRefVec _outputDeltas;

    SnnVal (*_activation)(SnnVal)           = snn::logSigmoid;
    SnnVal (*_activationDerivative)(SnnVal) = snn::logSigmoidDerivative;

    BasicNeuron(SnnVal &value,
                SnnValVec &weights,
                SnnVal & (*activation)(SnnVal),
                SnnVal & (*activationDerivative)(SnnVal))
        : _value(value),
          _bias(0.0)
    {
        _inputWeights.insert(_inputWeights.begin(),
                             weights.begin() + 1,
                             weights.end());
        _bias = weights.front();
        _activation = activation;
        _activationDerivative = activationDerivative;
    }

    BasicNeuron &operator=(SnnVal newValue) {
        _value = newValue;
        return *this;
    }

    virtual void forward()
    {
        _inputSum = std::inner_product(_inputWeights.begin(),
                                       _inputWeights.end(),
                                       _inputValues.begin(),
                                       0.0);
        _value = _activation(_inputSum + _bias);
    }

    virtual void backward()
    {
        _delta = std::inner_product(_outputWeights.begin(),
                                    _outputWeights.end(),
                                    _outputDeltas.begin(),
                                    0.0);
    }

    virtual size_t getNumWeights() {
        // Connections + bias
        return _inputWeights.size() + 1;
    }

    virtual SnnVal getValue() {
        return _value;
    }
};


} // namespace snn

#endif