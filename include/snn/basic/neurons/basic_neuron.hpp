#ifndef SNN_BASIC_NEURON_HPP
#define SNN_BASIC_NEURON_HPP

#include <memory>

#include <snn/types.hpp>
#include <snn/basic/neurons/activation_functions.hpp>

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

    ActivationFunction *_activation = new LogSigmoid();

    BasicNeuron(): _activation(new LogSigmoid()) {}

    BasicNeuron(SnnVal &value,
                ActivationFunction *activation)
        : _value(value),
          _activation(activation) {}


    ~BasicNeuron() {
        // TODO napraw
        // delete _activation;
        // _activation = nullptr;
    }

    virtual void connectSource(BasicNeuron &source,
                               const SnnVal &weight)
    {
        _inputValues.push_back(source._value);
        _inputWeights.push_back(weight);
    }


    virtual void connectSink(BasicNeuron &sink)
    {
        _outputWeights.push_back(sink._inputWeights.back());
        _outputDeltas.push_back(sink._delta);
    }

    BasicNeuron &operator=(const SnnVal &newValue) {
        _value = newValue;
        return *this;
    }

    virtual void forward()
    {
        _inputSum = std::inner_product(_inputWeights.begin(),
                                       _inputWeights.end(),
                                       _inputValues.begin(),
                                       0.0);
        _value = _activation->value(_inputSum + _bias);
    }

    virtual void backward()
    {
        _delta = std::inner_product(_outputWeights.begin(),
                                    _outputWeights.end(),
                                    _outputDeltas.begin(),
                                    0.0);
        _delta *= _activation->derivative(_inputSum + _bias);
    }

    virtual void backward(const SnnVal &desiredOutput)
    {
        _delta = desiredOutput - _value;
        _delta *= _activation->derivative(_inputSum + _bias);
    }

    virtual size_t getNumWeights() {
        // Connections + bias
        return _inputWeights.size() + 1;
    }

    virtual SnnVal getValue() {
        return _value;
    }
};

template <typename... LearningParams>
SnnVal operator-(const SnnVal &value, const BasicNeuron<LearningParams...> &neuron) {
    return value - neuron._value;
}

} // namespace snn

#endif