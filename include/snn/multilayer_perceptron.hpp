#ifndef SNN_MULTILAYER_PERCEPTRON_HPP
#define SNN_MULTILAYER_PERCEPTRON_HPP



#include <initializer_list>
#include <iostream>
#include <vector>
#include <numeric>      // std::accumulate
#include <iterator>     // std::advance std::prev

namespace snn
{

/**
 * http://en.wikipedia.org/wiki/Multilayer_perceptron
 */
class MultilayerPerceptron
{
    struct Layer
    {
        std::vector<double>::iterator itValueBegin;
        std::vector<double>::iterator itValueEnd;
        std::vector<double>::iterator itWeightBegin;
        std::vector<double>::iterator itWeightEnd;
        double (*activation)(double);
        double (*activationDerivative)(double);
    };

    std::vector<double> _deltas;
    std::vector<double> _values;
    std::vector<double> _weights;
    std::vector<Layer>  _layers;

    size_t _inputDimmention;
    size_t _outputDimmention;
    size_t _numLayers;
    void computeLayerValues(Layer &prevLayer, Layer &layer);

public:
    MultilayerPerceptron(std::initializer_list<int> layer);
    MultilayerPerceptron(): MultilayerPerceptron({1, 3, 1}) {};

    std::vector<double> forward(std::vector<double> input);
    std::vector<double> backward(std::vector<double> input);

    // void addToWeights(std::vector<double> )

};

} // namespace snn

#endif