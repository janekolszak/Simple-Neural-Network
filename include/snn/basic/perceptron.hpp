// #ifndef SNN_PERCEPTRON_HPP
// #define SNN_PERCEPTRON_HPP

// #include <initializer_list>
// #include <iostream>
// #include <vector>
// #include <numeric>      // std::accumulate
// #include <iterator>     // std::advance std::prev
// #include <functional>   // std::reference_wrapper
// #include <chrono>
// #include <random>

// #include <snn/activation_functions.hpp>
// #include <snn/logging.hpp>
// #include <snn/types.hpp>
// #include <snn/basic/layer.hpp>
// #include <boost/ptr_container/ptr_vector.hpp>
// #include <boost/iterator/zip_iterator.hpp>
// #include <boost/tuple/tuple.hpp>

// namespace snn {


// typedef boost::ptr_vector<Layer> SnnLayerVec;

// struct Perceptron {
//     SnnLayerVec _layers;

//     Perceptron(std::initializer_list<int> layerSizes) {
//         // Setup layers
//         _layers.reserve(layerSizes.size());
//         for (int layerSize : layerSizes)
//             _layers.push_back(new snn::Layer(layerSize));


//         // Weights are going to be random
//         unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//         std::default_random_engine generator (seed);
//         std::uniform_real_distribution<SnnVal> distribution (-1.0, 1.0);

//         // Full connection between layers
//         for (auto itLayer = std::next(_layers.begin()),
//                 itPrevLayer = _layers.begin();
//                 itLayer != _layers.end();
//                 ++itLayer, ++itPrevLayer ) {
//             for (Neuron &sourceNeuron : itPrevLayer->_neurons)
//                 for (Neuron &sinkNeuron : itLayer->_neurons)
//                     snn::connectNeurons(sourceNeuron, sinkNeuron, distribution(generator));
//         }

//     }

//     void forward(std::initializer_list<SnnVal> inputsLst) {
//         SnnValVec inputsVec(inputsLst);
//         forward(inputsVec);
//     }

//     void forward(SnnValVec &inputs) {

//         assert(inputs.size() == _layers.front()._neurons.size());

//         _layers.front().set(inputs);

//         for (auto itLay = _layers.begin() + 1;
//                 itLay != _layers.end();
//                 ++itLay ) {
//             itLay->forward();
//         }


//         // for (auto &layer : _layers)
//         //     layer.forward();

//     }

//     SnnValVec getOutputs() {
//         SnnValVec outputs;
//         SnnNeuronVec &outNeurons = _layers.back()._neurons;

//         outputs.reserve(outNeurons.size());
//         for (Neuron &neuron : outNeurons)
//             outputs.push_back(neuron._value);

//         return outputs;
//     }

//     void backward(std::initializer_list<SnnVal> outputsLst) {
//         SnnValVec outputsVec(outputsLst);
//         backward(outputsVec);
//     }

//     void backward(SnnValVec &desiredOutputs) {
//         SnnNeuronVec &outNeurons = _layers.back()._neurons;

//         assert(desiredOutputs.size() == outNeurons.size());

//         // Compute difference between desired output
//         // and perception's output
//         auto b = boost::make_zip_iterator(boost::make_tuple(outNeurons.begin(), desiredOutputs.begin()));
//         auto e = boost::make_zip_iterator(boost::make_tuple(outNeurons.end(),   desiredOutputs.end()));

//         std::for_each(b, e, [](const boost::tuple<Neuron &, SnnVal &> &t)
//         {
//             t.get<0>()._delta = t.get<1>() - t.get<0>()._value;
//         });

//         for (auto itLay = _layers.rbegin() + 1;
//                 itLay != _layers.rend();
//                 ++itLay ) {
//             itLay->backward();
//         }
//         // for (auto &layer : _layers)
//         //     layer.backward();
//     }

//     void modify(SnnValVec beta) {
//         auto itBeta = beta.begin();
//         for (auto &layer : _layers)
//             layer.modify(itBeta);
//     }

//     // TODO: test
//     size_t getNumWeights() {
//         size_t numWeights = 0;

//         for (Layer &layer : _layers)
//             for (Neuron &neuron : layer._neurons)
//                 numWeights += neuron.getNumWeights();

//         return numWeights;
//     }

//     // TODO: test
//     SnnValVec getWeights() {
//         SnnValVec weights;
//         weights.reserve(getNumWeights());

//         auto b = std::next(_layers.begin());
//         for (auto itLay = b; itLay != _layers.end(); ++itLay) {
//             for (Neuron &neuron : itLay->_neurons)
//                 for (SnnVal &weight : neuron._inputWeights)
//                     weights.push_back(weight);
//         }

//         return weights;
//     }
// };

// } // namespace snn

// #endif