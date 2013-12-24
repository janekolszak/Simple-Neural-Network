#include <snn/basic/neurons/activation_functions.hpp>
#include <cmath>

namespace snn {

SnnVal logSigmoid(SnnVal x)
{
    return 1.0 /  (1.0 + std::exp(-1 * x));
}

SnnVal logSigmoidDerivative(SnnVal x)
{
    SnnVal logSigmoid =  1.0 / (1.0 + std::exp(-1 * x));
    return logSigmoid * (1.0 - logSigmoid);
}

SnnVal Identity(SnnVal x)
{
    return x;
}

SnnVal IdentityDerivative(SnnVal x)
{
    return 1;
}

}