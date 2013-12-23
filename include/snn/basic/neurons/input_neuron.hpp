#ifndef SNN_INPUT_NEURON_HPP
#define SNN_INPUT_NEURON_HPP

#include <snn/basic/neurons/basic_neuron.hpp>
#include <snn/types.hpp>

namespace snn {

template <typename... LearningParams>
struct InputNeuron: public BasicNeuron<LearningParams...> {

    InputNeuron(SnnVal value): BasicNeuron<LearningParams...>(value) {}
    InputNeuron(): BasicNeuron<LearningParams...>(0.0) {}

    virtual void forward() {};
    virtual void backward() {};
    virtual void learn(LearningParams... learningParams) {}

    size_t getNumWeights() {
        return 0;
    }

};

} // namespace snn

#endif