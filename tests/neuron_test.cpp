
#define BOOST_TEST_MODULE SNN
#include <boost/test/unit_test.hpp>

#include <snn/snn.hpp>
#include <vector>
#include <algorithm>
#include "test_utils.hpp"

using namespace snn;
using namespace std;


BOOST_AUTO_TEST_SUITE(VariousTests)

BOOST_AUTO_TEST_CASE( VARIOUS_SnnValVec_equal )
{
    SnnValVec a = {1, 2, 3};
    SnnValVec b = {1, 2, 3};

    BOOST_REQUIRE( a == b );
    BOOST_REQUIRE( !( a != b ));

    a = {1, 2, 3};
    b = {3, 2, 1};

    BOOST_REQUIRE( !( a == b ));
    BOOST_REQUIRE( a != b );

    a = {1, 2, 3};
    b = {1, 2, 3, 4};

    BOOST_REQUIRE( !( a == b ));
    BOOST_REQUIRE( a != b );
}


BOOST_AUTO_TEST_CASE( VARIOUS_activation_functions )
{
    LogSigmoid *ls = new LogSigmoid();
    BOOST_CHECK_CLOSE(ls->value(0.0), 0.5, 1);
    BOOST_CHECK_CLOSE(ls->derivative(0.0) , 0.25, 1);

    Linear *l = new Linear(0.0, 10.0, 9.0);
    BOOST_CHECK_CLOSE(l->value(1.0), 9.0, 1);
    BOOST_CHECK_CLOSE(l->derivative(1.0) , 9.0, 1);
}


BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(NeuronTest)

BOOST_AUTO_TEST_CASE( NEURON_forward )
{
    ScalarLearningRateNeuron n1, n2;

    n1._value = 1.;
    n1._activation = new LogSigmoid();

    n2._value = 2.;
    n2._activation = new LogSigmoid();

    connectNeurons(n1, n2, 1);

    n2.forward();

    BOOST_REQUIRE( n1._value == 1. );
    BOOST_REQUIRE( n2._value != 0. );
}


BOOST_AUTO_TEST_CASE( NEURON_backward )
{
    ScalarLearningRateNeuron n1, n2;

    n1._delta = 0.6;
    n2._delta = 1;

    connectNeurons(n1, n2, 1);

    n1.backward();

    BOOST_REQUIRE( n1._delta != 0.6 );
    BOOST_REQUIRE( n2._delta == 1 );
}


BOOST_AUTO_TEST_CASE( NEURON_learn )
{
    ScalarLearningRateNeuron n1, n2;

    n1._value = 1;
    n2._value = 0;
    n2._delta = 1;
    n2._activation = new snn::LogSigmoid;

    connectNeurons(n1, n2, 1);

    // Learning rate vector
    SnnVal learningRate = 0.25;

    n2.learn(learningRate);
    SnnVal weight = n2._inputWeights.front();
    BOOST_REQUIRE( weight != 1);

    n2.learn(learningRate);
    BOOST_REQUIRE( n2._inputWeights.front() != weight);
}

BOOST_AUTO_TEST_SUITE_END()




BOOST_AUTO_TEST_SUITE(LayerTest)

BOOST_AUTO_TEST_CASE( LAYER_forward )
{
    BasicLayer<ScalarLearningRateNeuron, SnnVal> l1 = {1, 1, 1};
    BasicLayer<ScalarLearningRateNeuron, SnnVal> l2 = {9, 9};

    for (auto &n1 : l1._neurons)
        for (auto &n2 : l2._neurons)
            connectNeurons(n1, n2, 1);

    l2.forward();

    for (auto &value : l2.getValues())
        BOOST_REQUIRE( value != 9);
}


BOOST_AUTO_TEST_CASE( LAYER_backward )
{
    BasicLayer<ScalarLearningRateNeuron, SnnVal> l1(4, new LogSigmoid);
    for (auto &n : l1._neurons) n._delta = 13.4;
    BasicLayer<ScalarLearningRateNeuron, SnnVal> l2(2, new LogSigmoid);
    for (auto &n : l2._neurons) n._delta = 0.2;

    for (auto &n1 : l1._neurons)
        for (auto &n2 : l2._neurons)
            connectNeurons(n1, n2, 1);

    l1.backward();

    for (auto &delta : l1.getDeltas())
        BOOST_REQUIRE( delta != 13.4);

}


BOOST_AUTO_TEST_CASE( LAYER_learn )
{
    BasicLayer<ScalarLearningRateNeuron, SnnVal> l1 = {1, 1, 1, 1};
    BasicLayer<ScalarLearningRateNeuron, SnnVal> l2 = {4};
    for (auto &n : l2._neurons) n._delta = 1;

    for (auto &n1 : l1._neurons)
        for (auto &n2 : l2._neurons)
            connectNeurons(n1, n2, 1);

    // Learning rate vector
    SnnVal learningRate = 2;

    l2.learn(learningRate);

    for (auto &n : l2._neurons)
        for (auto &weight : n._inputWeights)
            BOOST_REQUIRE( weight != 1);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(PerceptronTest)

BOOST_AUTO_TEST_CASE( PERCEPTRON_forward )
{
    ScalarLearningRatePerceptron p = {2, 2, 2};
    p.forward({1, 2});
}


BOOST_AUTO_TEST_CASE( PERCEPTRON_backward )
{
    ScalarLearningRatePerceptron p = {2, 2, 2};
    p.backward({1.4, 2});
}


BOOST_AUTO_TEST_CASE( PERCEPTRON_modify )
{
    ScalarLearningRatePerceptron p = {2, 10, 2};
    SnnVal learningRate = 1;

    p.forward({1.4, 2});
    p.backward({1.4, 2});
    p.learn(learningRate);
    SnnValVec out1 =  p.getOutputs();

    p.forward({1.4, 2});
    p.backward({1.4, 2});
    p.learn(learningRate);
    SnnValVec out2 =  p.getOutputs();

    BOOST_REQUIRE( out1 != out2);
}


BOOST_AUTO_TEST_CASE( PERCEPTRON_simple_backprop )
{
    SnnVal target = 0.415;

    ScalarLearningRatePerceptron p = {1, 1, 1, 1, 1, 1};
    SnnDataset trainSet = {
        {1.0}, { target}
    };

    train(p, trainSet, 10);

    // BOOST_CHECK_CLOSE( p.getOutputs().front(), target, 10 );
    testIfLearned(p, trainSet, 10);

    // printTrainingResults(p, trainSet);

}

BOOST_AUTO_TEST_CASE( PERCEPTRON_xor )
{
    ScalarLearningRatePerceptron p = {2, 3, 1};

    SnnDataset trainSet = {
        {0, 0}, {0},
        {1, 0}, {1},
        {0, 1}, {1},
        {1, 1}, {0}
    };

    train(p, trainSet, 10000, 0.3);

    // printTrainingResults(p, trainSet);
    testIfLearned(p, trainSet, 0.1);
}

BOOST_AUTO_TEST_CASE( ScalarLearningRateMomentumPerceptron_xor )
{
    ScalarLearningRateMomentumPerceptron p = {2, 3, 1};

    SnnDataset trainSet = {
        {0, 0}, {0},
        {1, 0}, {1},
        {0, 1}, {1},
        {1, 1}, {0}
    };

    train(p, trainSet, 100000, 0.3, 0.8);

    printTrainingResults(p, trainSet);
    testIfLearned(p, trainSet, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()

