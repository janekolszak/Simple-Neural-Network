#ifndef SNN_BASIC_NEURON_HPP
#define SNN_BASIC_NEURON_HPP

#include <snn/types.hpp>

namespace snn {

template <typename... LearningParams>
struct BasicNeuron {

    SnnVal _value = 0.0;

    BasicNeuron &operator=(SnnVal newValue) {
        _value = newValue;
        return *this;
    }

    BasicNeuron(SnnVal value): _value(value) {}
    BasicNeuron(): BasicNeuron(0.0) {}

    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void learn(LearningParams... learningParams) = 0;
    virtual size_t getNumWeights() = 0;
    virtual SnnVal getValue() {
        return _value;
    }
};

} // namespace snn

#endif