// #ifndef SNN_NEURON_HPP
// #define SNN_NEURON_HPP

// #include <initializer_list>
// #include <iostream>
// #include <vector>
// #include <numeric>      // std::accumulate
// #include <iterator>     // std::advance std::prev
// #include <functional>   // std::reference_wrapper

// #include <snn/activation_functions.hpp>
// #include <snn/logging.hpp>
// #include <snn/types.hpp>


// namespace snn {
// using namespace std; // TODO: delete
// struct Neuron {

//     friend void connectNeurons();

//     SnnVal _inputSum = 0.0; ///< linear sum of inputs
//     SnnVal _value = 0.0; ///< _activation(_inputSum)
//     SnnVal _bias = 0.0; ///< linear sum of inputs
//     SnnVal _delta = 0.0;
//     SnnValVec _inputWeights;
//     SnnValRefVec _outputWeights;
//     SnnValRefVec _inputValues;
//     SnnValRefVec _outputDeltas;

//     SnnVal (*_activation)(SnnVal) = snn::logSigmoid;
//     SnnVal (*_activationDerivative)(SnnVal) = snn::logSigmoidDerivative;

//     Neuron &operator=(SnnVal newValue) {
//         _value = newValue;
//         return *this;
//     }


//     Neuron(SnnVal value): _value(value), _bias(0.0) {}
//     Neuron(): Neuron(0.0) {}

//     void forward()
//     {
//         _inputSum = std::inner_product(_inputWeights.begin(),
//                                        _inputWeights.end(),
//                                        _inputValues.begin(),
//                                        0.0);
//         _value = _activation(_inputSum + _bias);
//     }

//     void backward()
//     {
//         _delta = std::inner_product(_outputWeights.begin(),
//                                     _outputWeights.end(),
//                                     _outputDeltas.begin(),
//                                     0.0);
//     }

//     void modify(SnnValVec::iterator itBeta)
//     {
//         SnnVal valueDerivative = _activationDerivative(_inputSum);

//         for (auto &weight : _inputWeights) {
//             weight += (*itBeta) * _delta *  valueDerivative ;
//             ++itBeta;
//         }
//         _bias += (*itBeta) * _delta *  valueDerivative;
//         ++itBeta;
//     }

//     size_t getNumWeights() {
//         // Connections + bias
//         return _inputWeights.size() + 1;
//     }
// };


// /**
//  * Function for connecting two neurons.
//  * Call it only once per pair.
//  *
//  * @param source neuron that generates the signal
//  * @param sink   neuron that recieves the signal
//  * @param weight weight of the connection
//  */
// void connectNeurons(Neuron &source, Neuron &sink, SnnVal weight) {
//     sink._inputValues.push_back(source._value);
//     sink._inputWeights.push_back(weight);
//     source._outputWeights.push_back(sink._inputWeights.back());
//     source._outputDeltas.push_back(sink._delta);
// }


// SnnVal operator-(const SnnVal &value, Neuron &neuron) {
//     return value - neuron._value;
// }



// } // namespace snn

// #endif