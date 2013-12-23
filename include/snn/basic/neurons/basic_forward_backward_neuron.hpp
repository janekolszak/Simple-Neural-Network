#ifndef SNN_BASIC_FORWARD_BACKWARD_NEURON_HPP
#define SNN_BASIC_FORWARD_BACKWARD_NEURON_HPP

#include <snn/basic/neurons/activation_functions.hpp>
#include <snn/basic/neurons/basic_neuron.hpp>
#include <snn/types.hpp>

namespace snn {

template <typename... LearningParams>
struct BasicForwardBackwardNeuron:  BasicNeuron<LearningParams...> {

    SnnVal _inputSum = 0.0; ///< linear sum of inputs
    SnnVal _bias = 0.0;
    SnnVal _delta = 0.0;
    SnnValVec _inputWeights;
    SnnValRefVec _outputWeights;
    SnnValRefVec _inputValues;
    SnnValRefVec _outputDeltas;

    SnnVal (*_activation)(SnnVal) = snn::logSigmoid;
    SnnVal (*_activationDerivative)(SnnVal) = snn::logSigmoidDerivative;

    BasicForwardBackwardNeuron(): BasicNeuron<LearningParams...>(0.0) {}

    BasicForwardBackwardNeuron(SnnVal value,
                               SnnVal (*activation)(SnnVal),
                               SnnVal (*activationDerivative)(SnnVal))
        : BasicNeuron<LearningParams...>(value),
          _bias(0.0)
    {
        _activation = activation;
        _activationDerivative = activationDerivative;
    }


    virtual void forward()
    {
        _inputSum = std::inner_product(_inputWeights.begin(),
                                       _inputWeights.end(),
                                       _inputValues.begin(),
                                       0.0);
        BasicNeuron<LearningParams...>::_value = _activation(_inputSum + _bias);
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

};



} // namespace snn

#endif