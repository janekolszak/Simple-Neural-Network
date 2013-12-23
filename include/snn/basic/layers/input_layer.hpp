#ifndef SNN_INPUT_LAYER_HPP
#define SNN_INPUT_LAYER_HPP

// #include <snn/basic/neurons/input_neuron.hpp>
#include <snn/basic/layers/basic_layer.hpp>

namespace snn {

template <typename NeuronType, typename... LearningParams>
struct InputLayer: public BasicLayer<NeuronType, LearningParams...> {

    InputLayer(size_t numNeurons):
        BasicLayer<NeuronType, LearningParams...>(numNeurons) {}

    InputLayer(std::initializer_list<SnnVal> values,
               SnnVal & (*activation)(SnnVal),
               SnnVal & (*activationDerivative)(SnnVal) ):
        BasicLayer<NeuronType, LearningParams...>(values, activation, activationDerivative) {}

    virtual void forward() { }
    virtual void backward() { }
    virtual void learn(LearningParams... learningParams) {}

};

} // namespace snn

#endif