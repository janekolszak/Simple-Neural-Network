#ifndef SNN_INPUT_LAYER_HPP
#define SNN_INPUT_LAYER_HPP

// #include <snn/basic/neurons/input_neuron.hpp>
#include <snn/basic/layers/basic_forward_backward_layer.hpp>

namespace snn {

template <typename NeuronType, typename... LearningParams>
struct InputLayer: public BasicForwardBackwardLayer<NeuronType, LearningParams...> {

    InputLayer(size_t numNeurons):
        BasicForwardBackwardLayer<NeuronType, LearningParams...>(numNeurons) {}
    InputLayer(std::initializer_list<SnnVal> values):
        BasicForwardBackwardLayer<NeuronType, LearningParams...>(values) {}

    virtual void forward() { }
    virtual void backward() { }
    virtual void learn(LearningParams... learningParams) {}
    virtual SnnValVec deltas() {
        SnnValVec v;
        return v;
    }
};

} // namespace snn

#endif