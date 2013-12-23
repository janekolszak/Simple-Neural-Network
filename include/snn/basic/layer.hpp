// #ifndef SNN_LAYER_HPP
// #define SNN_LAYER_HPP

// #include <initializer_list>
// #include <iostream>
// #include <vector>
// #include <numeric>      // std::accumulate
// #include <iterator>     // std::advance std::prev
// #include <functional>   // std::reference_wrapper
// #include <assert.h>

// #include <snn/activation_functions.hpp>
// #include <snn/logging.hpp>
// #include <snn/types.hpp>
// #include <snn/basic/neuron.hpp>
// #include <boost/ptr_container/ptr_vector.hpp>
// #include <boost/iterator/zip_iterator.hpp>
// #include <boost/tuple/tuple.hpp>

// namespace snn {

// typedef boost::ptr_vector<Neuron> SnnNeuronVec;

// /**
//  * Layer stores neurons, that are independent from each other.
//  *
//  * TODO: The for loops can be concurent.
//  */
// struct Layer {
//     Trzeba zrobic input Layer zeby nie bylo biasa na wyjsciu.
//     W tym celu porozdzielaj juz na klasy potomne i pliki do src.

//     SnnNeuronVec _neurons;

//     Layer(size_t numNeurons) {
//         _neurons.reserve(numNeurons);
//         for (size_t i = 0; i < numNeurons; ++i)
//             _neurons.push_back(new snn::Neuron);
//     }

//     Layer(std::initializer_list<SnnVal> values) {
//         _neurons.reserve(values.size());
//         for (SnnVal value : values)
//             _neurons.push_back(new snn::Neuron(value));
//     }

//     void forward() {
//         for (auto &neuron : _neurons)
//             neuron.forward();
//     }

//     void backward() {
//         for (auto &neuron : _neurons)
//             neuron.backward();
//     }

//     void modify(SnnValVec::iterator itBeta) {
//         for (auto &neuron : _neurons)
//             neuron.modify(itBeta);
//     }

//     SnnValVec values() {
//         SnnValVec values;
//         values.reserve(_neurons.size());
//         for (auto &neuron : _neurons)
//             values.push_back(neuron._value);
//         return values;
//     }

//     SnnValVec deltas() {
//         SnnValVec deltas;
//         deltas.reserve(_neurons.size());
//         for (auto &neuron : _neurons)
//             deltas.push_back(neuron._delta);
//         return deltas;
//     }

//     void set(SnnValVec &values) {
//         assert(values.size() == _neurons.size());
//         std::copy(values.begin(), values.end(), _neurons.begin());
//     }

// };

// } // namespace snn

// #endif