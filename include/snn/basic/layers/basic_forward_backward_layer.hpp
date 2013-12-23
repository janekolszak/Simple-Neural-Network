#ifndef SNN_HIDDEN_LAYER_HPP
#define SNN_HIDDEN_LAYER_HPP

#include <type_traits>  // std::is_base_of
#include <initializer_list>
#include <assert.h>
#include <boost/ptr_container/ptr_vector.hpp>

#include <snn/types.hpp>
#include <snn/basic/neurons/basic_forward_backward_neuron.hpp>

namespace snn {

template <typename NeuronType, typename... LearningParams>
struct BasicForwardBackwardLayer
        : BasicLayer<NeuronType, LearningParams...> {

    static_assert(std::is_base_of<BasicForwardBackwardNeuron<LearningParams...>, NeuronType>::value,
                  "BasicForwardBackwardNeuron needs to be a base class of NeuronType");

    typedef boost::ptr_vector<BasicNeuron<LearningParams...> > SnnNeuronVec;
    typedef NeuronType neurontype;
    SnnNeuronVec _neurons;

    BasicForwardBackwardLayer(size_t numNeurons)
        : BasicLayer<NeuronType, LearningParams...>(numNeurons) { }

    BasicForwardBackwardLayer(std::initializer_list<SnnVal> values)
        : BasicLayer<NeuronType, LearningParams...>(values) {}

    virtual void forward() {
        for (auto &neuron : _neurons)
            neuron.forward();
    }

    virtual void backward() {
        for (auto &neuron : _neurons)
            neuron.backward();
    }

    virtual void learn(LearningParams... learningParams) {
        for (auto &neuron : _neurons)
            neuron.learn(learningParams...);
    }

    virtual SnnValVec deltas() {
        SnnValVec deltas;
        deltas.reserve(_neurons.size());
        for (auto &neuron : _neurons)
            deltas.push_back(neuron._delta);
        return deltas;
    }

};

} // namespace snn

#endif