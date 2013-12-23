#ifndef SNN_BASIC_LAYER_HPP
#define SNN_BASIC_LAYER_HPP

#include <type_traits>  // std::is_base_of
#include <initializer_list>
#include <assert.h>
#include <boost/ptr_container/ptr_vector.hpp>

#include <snn/types.hpp>
#include <snn/basic/neurons/basic_neuron.hpp>
#include <snn/basic/neurons/activation_functions.hpp>

namespace snn {


/**
 * Layer stores neurons, that are independent from each other.
 *
 * TODO: The for loops can be concurent.
 */
// TODO: Get learning Params from NeuronType
template <typename NeuronType, typename... LearningParams>
struct BasicLayer {

    static_assert(std::is_base_of<BasicNeuron<LearningParams...>, NeuronType>::value,
                  "BasicNeuron needs to be a base class of NeuronType");

    typedef boost::ptr_vector<NeuronType> neuron_vec;
    typedef NeuronType neuron_type;

    neuron_vec _neurons;

    BasicLayer(size_t numNeurons,
               SnnVal (*activation)(SnnVal),
               SnnVal (*activationDerivative)(SnnVal)) {
        _neurons.reserve(numNeurons);
        for (size_t i = 0; i < numNeurons; ++i)
            _neurons.push_back(new NeuronType(0.0, activation, activationDerivative));
    }

    BasicLayer(std::initializer_list<SnnVal> values ) {
        // _neurons.reserve(values.size());
        for (SnnVal value : values)
            _neurons.push_back(new NeuronType(value, snn::logSigmoid, snn::logSigmoidDerivative));
    }

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

    virtual SnnValVec getDeltas() {
        SnnValVec deltas;
        deltas.reserve(_neurons.size());
        for (auto &neuron : _neurons)
            deltas.push_back(neuron._delta);
        return deltas;
    }

    virtual void setValues(SnnValVec &values) {
        assert(values.size() == _neurons.size());
        std::copy(values.begin(), values.end(), _neurons.begin());
    }

    virtual SnnValVec getValues() {
        SnnValVec values;
        values.reserve(_neurons.size());
        for (auto &neuron : _neurons)
            values.push_back(neuron._value);
        return values;
    }

    virtual neuron_vec &neurons() {
        return _neurons;
    }

};

} // namespace snn

#endif