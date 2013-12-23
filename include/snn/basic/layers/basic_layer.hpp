#ifndef SNN_BASIC_LAYER_HPP
#define SNN_BASIC_LAYER_HPP

#include <type_traits>  // std::is_base_of
#include <initializer_list>
#include <assert.h>
#include <boost/ptr_container/ptr_vector.hpp>

#include <snn/types.hpp>
#include <snn/basic/neurons/basic_neuron.hpp>

namespace snn {


/**
 * Layer stores neurons, that are independent from each other.
 *
 * TODO: The for loops can be concurent.
 */
template <typename NeuronType, typename... LearningParams>
struct BasicLayer {

    // static_assert(std::is_base_of<BasicNeuron<LearningParams...>, NeuronType>::value,
    //               "BasicNeuron needs to be a base class of NeuronType");

    typedef boost::ptr_vector<NeuronType> SnnNeuronVec;
    typedef NeuronType neurontype;

    SnnNeuronVec _neurons;

    BasicLayer(size_t numNeurons) {
        _neurons.reserve(numNeurons);
        for (size_t i = 0; i < numNeurons; ++i)
            _neurons.push_back(new NeuronType);
    }

    BasicLayer(std::initializer_list<SnnVal> values) {
        _neurons.reserve(values.size());
        for (SnnVal value : values)
            _neurons.push_back(new NeuronType(value));
    }

    virtual SnnValVec values() {
        SnnValVec values;
        values.reserve(_neurons.size());
        for (auto &neuron : _neurons)
            values.push_back(neuron._value);
        return values;
    }

    virtual void setValues(SnnValVec &values) {
        assert(values.size() == _neurons.size());
        std::copy(values.begin(), values.end(), _neurons.begin());
    }

    virtual void forward() = 0;

    virtual void backward() = 0;

    virtual void learn(LearningParams... learningParams) = 0;

    virtual SnnValVec deltas() = 0;

};

} // namespace snn

#endif