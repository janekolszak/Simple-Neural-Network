#ifndef SNN_OUTPUT_LAYER_HPP
#define SNN_OUTPUT_LAYER_HPP

#include <snn/basic/layers/basic_layer.hpp>

namespace snn {

template <typename NeuronType, typename... LearningParams>
struct OutputLayer : public BasicLayer<NeuronType, LearningParams...> {

    OutputLayer(size_t numNeurons,
                SnnVal (*activation)(SnnVal),
                SnnVal (*activationDerivative)(SnnVal)):
        BasicLayer<NeuronType, LearningParams...>(numNeurons, activation, activationDerivative) {}

    OutputLayer(std::initializer_list<SnnVal> values):
        BasicLayer<NeuronType, LearningParams...>(values) {}

    virtual void backward() {}

};

} // namespace snn

#endif