#ifndef SNN_OUTPUT_LAYER_HPP
#define SNN_OUTPUT_LAYER_HPP

#include <snn/basic/layers/basic_forward_backward_layer.hpp>

namespace snn {

template <typename NeuronType, typename... LearningParams>
struct OutputLayer : public BasicForwardBackwardLayer<NeuronType, LearningParams...> {

    OutputLayer(size_t numNeurons):
        BasicForwardBackwardLayer<NeuronType, LearningParams...>(numNeurons) {}
    OutputLayer(std::initializer_list<SnnVal> values):
        BasicForwardBackwardLayer<NeuronType, LearningParams...>(values) {}

    virtual void backward() {}

};

} // namespace snn

#endif