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
    sink.connectSource(source, weight);
    source.connectSink(sink);
}

} // namespace snn

#endif