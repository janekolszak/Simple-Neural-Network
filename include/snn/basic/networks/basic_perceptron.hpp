#ifndef SNN_BASIC_PERCEPTRON_HPP
#define SNN_BASIC_PERCEPTRON_HPP

#include <initializer_list>
#include <iterator>     // std::advance std::prev
#include <chrono>
#include <random>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>


#include <snn/types.hpp>
#include <snn/basic/layers/basic_layer.hpp>
#include <snn/basic/layers/input_layer.hpp>
#include <snn/basic/layers/output_layer.hpp>
#include <snn/basic/layers/basic_forward_backward_layer.hpp>


namespace snn {

template <typename NeuronType, typename... LearningParams>
struct BasicPerceptron {
    typedef boost::ptr_vector<BasicLayer<NeuronType, LearningParams...> > SnnLayerVec;
    typedef typename BasicLayer<NeuronType, LearningParams...>::SnnNeuronVec SnnNeuronVec;
    SnnLayerVec _layers;

    BasicPerceptron() {};
    BasicPerceptron(std::initializer_list<size_t> layerSizes) {
        assert(layerSizes.size() >= 2);

        // Setup layers
        _layers.reserve(layerSizes.size());
        _layers.push_back(new InputLayer<NeuronType, LearningParams...>(*layerSizes.begin()));

        for (auto itLayer = _layers.begin() + 1;
                itLayer != _layers.end() - 1;
                ++itLayer)
            _layers.push_back(new BasicForwardBackwardLayer<NeuronType, LearningParams...>(*itLayer));

        _layers.push_back(new OutputLayer<NeuronType, LearningParams...>(*(_layers.end() - 1)));

        // Weights are going to be random
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::uniform_real_distribution<SnnVal> distribution (-1.0, 1.0);

        // Full connection between layers
        for (auto itLayer = std::next(_layers.begin()),
                itPrevLayer = _layers.begin();
                itLayer != _layers.end();
                ++itLayer, ++itPrevLayer ) {
            for (auto &sourceNeuron : itPrevLayer->_neurons)
                for (auto &sinkNeuron : itLayer->_neurons)
                    snn::connectNeurons(sourceNeuron, sinkNeuron, distribution(generator));
        }

    }

    void forward(std::initializer_list<SnnVal> inputsLst) {
        SnnValVec inputsVec(inputsLst);
        forward(inputsVec);
    }

    void forward(SnnValVec &inputs) {

        assert(inputs.size() == _layers.front()._neurons.size());

        _layers.front().setValues(inputs);

        for (auto &layer : _layers)
            layer.forward();
    }

    SnnValVec getOutputs() {
        SnnValVec outputs;
        SnnNeuronVec &outNeurons = _layers.back()._neurons;

        outputs.reserve(outNeurons.size());
        for (auto &neuron : outNeurons)
            outputs.push_back(neuron._value);

        return outputs;
    }

    void backward(std::initializer_list<SnnVal> outputsLst) {
        SnnValVec outputsVec(outputsLst);
        backward(outputsVec);
    }

    void backward(SnnValVec &desiredOutputs) {
        SnnNeuronVec &outNeurons = _layers.back()._neurons;

        assert(desiredOutputs.size() == outNeurons.size());

        // Compute difference between desired output
        // and perception's output
        auto b = boost::make_zip_iterator(boost::make_tuple(outNeurons.begin(), desiredOutputs.begin()));
        auto e = boost::make_zip_iterator(boost::make_tuple(outNeurons.end(),   desiredOutputs.end()));

        // TODO odkomentuj
        // std::for_each(b, e, [](const boost::tuple<NeuronType &, SnnVal &> &t)
        // {
        //     t.get<0>()._delta = t.get<1>() - t.get<0>()._value;
        // });

        for (auto &layer : _layers)
            layer.backward();

    }

    // void learn(SnnValVec beta) {
    //     auto itBeta = beta.begin();
    //     for (auto &layer : _layers)
    //         layer.learn(itBeta);
    // }


    size_t getNumWeights() {
        size_t numWeights = 0;

        for (auto &layer : _layers)
            for (auto &neuron : layer._neurons)
                numWeights += neuron.getNumWeights();

        return numWeights;
    }

    SnnValVec getWeights() {
        SnnValVec weights;
        weights.reserve(getNumWeights());

        auto b = std::next(_layers.begin());
        for (auto itLay = b; itLay != _layers.end(); ++itLay) {
            for (auto &neuron : itLay->_neurons)
                for (SnnVal &weight : neuron._inputWeights)
                    weights.push_back(weight);
        }

        return weights;
    }
};

} // namespace snn

#endif