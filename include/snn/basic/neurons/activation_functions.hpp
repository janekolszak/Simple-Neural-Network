#ifndef SNN_ACTIVATION_FUNCTIONS_HPP
#define SNN_ACTIVATION_FUNCTIONS_HPP

#include <snn/types.hpp>

/**
 * Basically this:
 * http://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions
 *
 */

namespace snn {

SnnVal logSigmoid(SnnVal x);
SnnVal logSigmoidDerivative(SnnVal x);
SnnVal Identity(SnnVal x);
SnnVal IdentityDerivative(SnnVal x);

}

#endif