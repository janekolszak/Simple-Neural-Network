#ifndef SNN_OUTPUT_LAYER_HPP
#define SNN_OUTPUT_LAYER_HPP

#include <snn/basic/layers/basic_layer.hpp>

namespace snn {

template <typename NeuronType, typename... LearningParams>
struct OutputLayer : public BasicLayer<NeuronType, LearningParams...> {

    OutputLayer(size_t numNeurons,
                ActivationFunction *activation):
        BasicLayer<NeuronType, LearningParams...>(numNeurons, activation) {}

    OutputLayer(std::initializer_list<SnnVal> values):
        BasicLayer<NeuronType, LearningParams...>(values) {}

    virtual void backward() {}

};

} // namespace snn

#endif