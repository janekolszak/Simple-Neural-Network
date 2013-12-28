#ifndef SNN_INPUT_LAYER_HPP
#define SNN_INPUT_LAYER_HPP

// #include <snn/basic/neurons/input_neuron.hpp>
#include <snn/basic/layers/basic_layer.hpp>
#include <snn/basic/neurons/activation_functions.hpp>

namespace snn {

template <typename NeuronType, typename... LearningParams>
struct InputLayer: public BasicLayer<NeuronType, LearningParams...> {

    InputLayer(size_t numNeurons,
               ActivationFunction *activation):
        BasicLayer<NeuronType, LearningParams...>(numNeurons, activation) {}

    InputLayer(std::initializer_list<SnnVal> values):
        BasicLayer<NeuronType, LearningParams...>(values) {}

    virtual void forward() { }
    virtual void backward() { }
    virtual void learn(LearningParams... learningParams) {}

};

} // namespace snn

#endif