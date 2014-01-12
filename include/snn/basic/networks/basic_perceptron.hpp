#ifndef SNN_BASIC_PERCEPTRON_HPP
#define SNN_BASIC_PERCEPTRON_HPP

#include <initializer_list>
#include <vector>
#include <iterator>     // std::advance std::prev
#include <chrono>
#include <iostream>
#include <random>
#include <boost/ptr_container/ptr_vector.hpp>

#include <boost/python/list.hpp>
#include <boost/python/stl_iterator.hpp>


#include <snn/types.hpp>
#include <snn/basic/layers/basic_layer.hpp>
#include <snn/basic/layers/input_layer.hpp>
#include <snn/basic/layers/output_layer.hpp>
#include <snn/basic/layers/basic_layer.hpp>
#include <snn/basic/neurons/activation_functions.hpp>

namespace snn {
// TODO delete
using namespace std;
template <typename NeuronType, typename... LearningParams>
struct BasicPerceptron {

    typedef boost::ptr_vector<BasicLayer<NeuronType, LearningParams...> > layer_vec;
    typedef typename BasicLayer<NeuronType, LearningParams...>::neuron_vec neuron_vec;
    layer_vec _layers;

    virtual void learn(LearningParams... learningParams) = 0;

    BasicPerceptron(std::initializer_list<size_t> layerSizes)
    {
        // Setup layers
        assert(layerSizes.size() >= 2);
        _layers.reserve(layerSizes.size());
        _initLayers(layerSizes.begin(), layerSizes.end(), 0.0, 1.0);

        _initWeights(0.0, 1.0);
    }

    BasicPerceptron(std::vector<size_t> &layerSizes, const SnnVal &outMin, const SnnVal &outMax)
    {
        // Setup layers
        assert(layerSizes.size() >= 2);
        _layers.reserve(layerSizes.size());
        _initLayers(layerSizes.begin(), layerSizes.end(), outMin, outMax);

        _initWeights(outMin, outMax);
    }

    BasicPerceptron(boost::python::object &layerSizes, const SnnVal &outMin, const SnnVal &outMax)
    {
        auto begin = boost::python::stl_input_iterator<size_t>(layerSizes);
        auto end   = boost::python::stl_input_iterator<size_t>();
        _initLayers(begin, end, outMin, outMax);
        assert(_layers.size() >= 2);
        _layers.shrink_to_fit();

        _initWeights(outMin, outMax);
    }

    template <class InputIterator>
    void _initLayers(const InputIterator &itBegin,
                     const InputIterator &itEnd,
                     const SnnVal &outMin,
                     const SnnVal &outMax)
    {
        _layers.push_back(new InputLayer<NeuronType, LearningParams...>
                          (*itBegin, new snn::LogSigmoid()));

        for (InputIterator itLayerSize = itBegin + 1;
                itLayerSize != itEnd - 1;
                ++itLayerSize)
            _layers.push_back(new BasicLayer<NeuronType, LearningParams...>
                              (*itLayerSize, new snn::LogSigmoid(outMin, outMax)));

        _layers.push_back(new OutputLayer<NeuronType, LearningParams...>
                          (*(itEnd - 1), new snn::LogSigmoid(outMin, outMax)));
    }

    void _initWeights(const SnnVal &outMin, const SnnVal &outMax)
    {
        // Weights are going to be random,
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::uniform_real_distribution<SnnVal> distribution (outMin, outMax);

        // Full connection between layers
        for (auto itLayer = std::next(_layers.begin()),
                itPrevLayer = _layers.begin();
                itLayer != _layers.end();
                ++itLayer, ++itPrevLayer ) {
            for (auto &sourceNeuron : itPrevLayer->neurons())
                for (auto &sinkNeuron : itLayer->neurons())
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

        for (auto itL = _layers.begin() + 1;
                itL != _layers.end();
                ++itL)
            itL->forward();
    }

    void backward(std::initializer_list<SnnVal> outputsLst) {
        SnnValVec outputsVec(outputsLst);
        backward(outputsVec);
    }

    void backward(SnnValVec &desiredOutputs) {
        auto &outNeurons = _layers.back()._neurons;

        assert(desiredOutputs.size() == outNeurons.size());

        // Compute difference between desired output
        // and perception's output
        auto itN = outNeurons.begin();
        auto itO = desiredOutputs.begin();
        for (; itO != desiredOutputs.end();
                ++itN, ++itO) {
            itN->backward(*itO);
        }

        for (auto itL = _layers.rbegin() + 1;
                itL != _layers.rend();
                ++itL)
            itL->backward();
    }

    SnnValVec getOutputs() {
        SnnValVec outputs;
        neuron_vec &outNeurons = _layers.back()._neurons;

        outputs.reserve(outNeurons.size());
        for (auto &neuron : outNeurons)
            outputs.push_back(neuron._value);

        return outputs;
    }

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

template <typename NeuronType, typename... LearningParams>
void printTrainingResults(BasicPerceptron<NeuronType, LearningParams... > &perceptron,
                          SnnDataset &trainSet) {
    for (auto itSet =  trainSet.begin();
            itSet != trainSet.end();
            std::advance (itSet, 2)) {
        SnnValVec &in = *itSet;
        SnnValVec &out = *std::next(itSet);

        perceptron.forward(in);
        std::cout << in << " " << out << " " << perceptron.getOutputs() << std::endl;
        std::cout.flush();
    }
}

} // namespace snn

#endif