#ifndef SNN_BASIC_NETWORK_HPP
#define SNN_BASIC_NETWORK_HPP

#include <initializer_list>

#include <snn/types.hpp>

namespace snn {

template <typename... LearningParams>
struct BasicNetwork {

    virtual void forward(std::initializer_list<SnnVal> inputsLst) {
        SnnValVec inputsVec(inputsLst);
        forward(inputsVec);
    }

    virtual void forward(SnnValVec &inputs) = 0;

    void backward(std::initializer_list<SnnVal> outputsLst) {
        SnnValVec outputsVec(outputsLst);
        backward(outputsVec);
    }

    virtual void backward(SnnValVec &desiredOutputs) = 0;

    void learn(LearningParams... learningParams) = 0;

    size_t getNumWeights() = 0;

    SnnValVec getWeights() = 0;

    SnnValVec getOutputs() = 0;
};

} // namespace snn

#endif