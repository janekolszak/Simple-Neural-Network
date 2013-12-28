#ifndef SNN_ACTIVATION_FUNCTIONS_HPP
#define SNN_ACTIVATION_FUNCTIONS_HPP

#include <snn/types.hpp>
#include <cmath>

/**
 * Basically this:
 * http://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions
 *
 */

namespace snn {

struct ActivationFunction {
    SnnVal _outMin = 0.0;
    SnnVal _outMax = 1.0;
    SnnVal _outScale = _outMax - _outMin;
    ActivationFunction() {}
    ActivationFunction(SnnVal outMin, SnnVal outMax):
        _outMin(outMin),
        _outMax(outMax),
        _outScale(_outMax - _outMin) {}
    virtual SnnVal value(const SnnVal &x) = 0;
    virtual SnnVal derivative(const SnnVal &x) = 0;
};

struct LogSigmoid: ActivationFunction {
    LogSigmoid(): ActivationFunction() {}
    LogSigmoid(SnnVal outMin, SnnVal outMax): ActivationFunction(outMin, outMax) {}

    SnnVal value(const SnnVal &x) {
        return _outMin + _outScale * 1.0 /  (1.0 + std::exp(-1 * x));
    }

    SnnVal derivative(const SnnVal &x) {
        SnnVal logSigmoid = 1.0 / (1.0 + std::exp(-1 * x));
        return _outScale * logSigmoid * (1.0 - logSigmoid);
    }
};

struct LinearScalingFunction: ActivationFunction {
    SnnVal _inpMin = 0.0;
    SnnVal _inpMax = 1.0;
    SnnVal _inpScale = 1.0;
    SnnVal _a;
    LinearScalingFunction(): ActivationFunction() {}
    LinearScalingFunction(const SnnVal &inpMin,
                          const SnnVal &inpMax,
                          const SnnVal &outMin,
                          const SnnVal &outMax)
        : ActivationFunction(outMin, outMax),
          _inpMin(inpMin),
          _inpMax(inpMax),
          _inpScale(inpMax - inpMin),
          _a(_outScale / _inpScale) {}

    SnnVal value(const SnnVal &x) {
        if (x < _inpMin)
            return _outMin;
        if (x > _inpMax )
            return _outMax;

        return _outMin + _a * (x - _inpMin);
    }

    SnnVal derivative(const SnnVal &x) {
        if (x < _inpMin || x > _inpMax )
            return 0.0;
        return _a;
    }
};

struct Linear: ActivationFunction {
    SnnVal _a;
    Linear(): ActivationFunction(0.0, 1.0), _a(1.0) {}
    Linear(SnnVal min, SnnVal max, SnnVal a): ActivationFunction(min, max), _a(a) {}

    SnnVal value(const SnnVal &x) {
        SnnVal val = _a * x;
        if (val < _outMin)
            return _outMin;

        if (val > _outMax)
            return _outMax;

        return val;
    }

    SnnVal derivative(const SnnVal &x) {
        SnnVal val = _a * x ;

        if (val < _outMin)
            return 0;

        if (val > _outMax)
            return 0;

        return _a;
    }
};

} // namespace snn

#endif