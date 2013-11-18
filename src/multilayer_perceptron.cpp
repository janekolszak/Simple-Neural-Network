#include <snn/multilayer_perceptron.hpp>

snn::MultilayerPerceptron::MultilayerPerceptron(std::initializer_list<int> layerSizes)
{
    // Reserve place for neuron's values
    int numNeurons = std::accumulate(layerSizes.begin(),
                                     layerSizes.end(),
                                     0);
    _values.resize(numNeurons);
    _values.shrink_to_fit();


    // Fill layers
    auto itValue = _values.begin();
    auto itWeight = _weights.begin();

    // Fill the input layer
    _layers.push_back(Layer());
    Layer &layer        = _layers.back();
    layer.itValueBegin  = itValue;
    std::advance(itValue, *layerSizes.begin());
    layer.itValueEnd    = itValue ;
    layer.itWeightBegin = itWeight;
    layer.itWeightEnd   = itWeight;

    // Fill the layers
    for (auto itLayerSize = layerSizes.begin() + 1; itLayerSize != layerSizes.end(); ++itLayerSize )
    {
        _layers.push_back(Layer());
        Layer &layer       = _layers.back();
        layer.itValueBegin = itValue;
        std::advance(itValue, *itLayerSize);
        layer.itValueEnd   = itValue;

        layer.itWeightBegin = itWeight ;
        std::advance(itWeight, *std::prev(itLayerSize));
        layer.itWeightEnd   = itWeight ;
    }

}

void
snn::MultilayerPerceptron::computeLayerValues(Layer &prevLayer,
        Layer &layer)
{

}

std::vector<double>
snn::MultilayerPerceptron::forward(std::vector<double> input)
{
    std::copy(input.begin(), input.end(), _values.begin());

    for (auto itLayer = _layers.begin() + 1; itLayer != _layers.end(); ++itLayer )
    {
        Layer &prevLayer = *std::prev(itLayer);
        Layer &layer     = *(itLayer);

        for (auto itValue = layer.itValueBegin; itValue !=  layer.itValueEnd; ++itValue)

            *itValue = layer.activation(std::inner_product(layer.itWeightBegin,
                                        layer.itWeightEnd,
                                        prevLayer.itValueBegin,
                                        0));

    }

    return std::vector<double>(_values.end() - _outputDimmention,
                               _values.end());
}


std::vector<double>
snn::MultilayerPerceptron::backward(std::vector<double> input)
{
    auto itFirsValueOfLastLayer = _values.end() - input.size();
    std::copy(input.begin(), input.end(), itFirsValueOfLastLayer);

    for (auto itLayer = _layers.rbegin() + 1; itLayer != _layers.rend(); ++itLayer)
    {
        // It's tricky, because we iterate backwards.
        Layer &prevLayer = *std::prev(itLayer);
        Layer &layer     = *(itLayer);

        for (auto itPrevValue = prevLayer.itValueBegin; itPrevValue !=  prevLayer.itValueEnd; ++itPrevValue)
        {
            auto itValue = layer.itValueBegin;
            for (auto itPrevWeight = prevLayer.itWeightBegin;
                    itPrevWeight !=  prevLayer.itWeightEnd;
                    ++itPrevWeight,
                    ++itValue)
            {
                *itValue += *itPrevValue * (*itPrevWeight);
            }
        }

    }

    return std::vector<double>(_values.begin(),
                               _values.begin() + _inputDimmention);
}