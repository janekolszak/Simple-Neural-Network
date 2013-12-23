#ifndef SNN_NEURON_UTILS_HPP
#define SNN_NEURON_UTILS_HPP

#include <snn/types.hpp>
#include <snn/basic/neurons/basic_neuron.hpp>

namespace snn {

/**
 * Function for connecting two neurons.
 * Call it only once per pair.
 *
 * @param source neuron that generates the signal
 * @param sink   neuron that recieves the signal
 * @param weight weight of the connection
 */
template<typename... LearningParams>
void connectNeurons(BasicNeuron<LearningParams...> &source,
                    BasicNeuron<LearningParams...> &sink,
                    SnnVal weight) {
    sink._inputValues.push_back(source._value);
    sink._inputWeights.push_back(weight);
    source._outputWeights.push_back(sink._inputWeights.back());
    source._outputDeltas.push_back(sink._delta);
}

} // namespace snn

#endif