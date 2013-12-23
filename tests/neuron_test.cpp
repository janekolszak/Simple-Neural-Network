
#define BOOST_TEST_MODULE SNN
#include <boost/test/unit_test.hpp>

#include <snn/snn.hpp>
#include <vector>
#include <algorithm>
#include <snn/implementations/scalar_learning_rate.hpp>

using namespace snn;
using namespace std;


BOOST_AUTO_TEST_SUITE(OperatorsTest)
BOOST_AUTO_TEST_CASE( PERCEPTRON_forward )
{
    // ScalarLearningRatePerceptron p = {2, 2, 2};
    // p.forward({1, 2});
}


BOOST_AUTO_TEST_CASE( PERCEPTRON_backward )
{
    // ScalarLearningRatePerceptron p = {2, 2, 2};
    // p.backward({1.4, 2});
}


BOOST_AUTO_TEST_SUITE_END()




// BOOST_AUTO_TEST_SUITE(OperatorsTest)
// BOOST_AUTO_TEST_CASE( SnnValVec_equal )
// {
//     SnnValVec a = {1, 2, 3};
//     SnnValVec b = {1, 2, 3};

//     BOOST_REQUIRE( a == b );
//     BOOST_REQUIRE( !( a != b ));

//     a = {1, 2, 3};
//     b = {3, 2, 1};

//     BOOST_REQUIRE( !( a == b ));
//     BOOST_REQUIRE( a != b );

//     a = {1, 2, 3};
//     b = {1, 2, 3, 4};

//     BOOST_REQUIRE( !( a == b ));
//     BOOST_REQUIRE( a != b );
// }


// BOOST_AUTO_TEST_SUITE_END()


// BOOST_AUTO_TEST_SUITE(NeuronTest)

// BOOST_AUTO_TEST_CASE( NEURON_forward )
// {
//     Neuron n1, n2;

//     n1._value = 1.;
//     n1._activation = snn::logSigmoid;

//     n2._value = 0.;
//     n2._activation = snn::logSigmoid;

//     connectNeurons(n1, n2, 1);

//     n2.forward();

//     BOOST_REQUIRE( n1._value == 1. );
//     BOOST_REQUIRE( n2._value != 0. );
// }


// BOOST_AUTO_TEST_CASE( NEURON_backward )
// {
//     Neuron n1, n2;

//     n1._delta = 0;
//     n2._delta = 1;

//     connectNeurons(n1, n2, 1);

//     n1.backward();

//     BOOST_REQUIRE( n1._delta != 0 );
//     BOOST_REQUIRE( n2._delta == 1 );
// }


// BOOST_AUTO_TEST_CASE( NEURON_modify)
// {
//     Neuron n1, n2;

//     n1._value = 1;
//     n2._value = 0;
//     n2._delta = 1;
//     n2._activationDerivative = snn::logSigmoidDerivative;

//     connectNeurons(n1, n2, 1);

//     // Learning rate vector
//     SnnValVec learningRate(1);
//     fill(learningRate.begin(), learningRate.end(), 1);

//     n2.modify(learningRate.begin());
//     SnnVal weight = n2._inputWeights.front();
//     BOOST_REQUIRE( weight != 1);

//     n2.modify(learningRate.begin());
//     BOOST_REQUIRE( n2._inputWeights.front() != weight);
// }

// BOOST_AUTO_TEST_SUITE_END()




// BOOST_AUTO_TEST_SUITE(LayerTest)

// BOOST_AUTO_TEST_CASE( LAYER_forward )
// {
//     Layer l1 = {1, 1, 1};
//     Layer l2 = {9, 9};

//     for (auto &n1 : l1._neurons)
//         for (auto &n2 : l2._neurons)
//             connectNeurons(n1, n2, 1);

//     l2.forward();

//     for (auto &value : l2.values())
//         BOOST_REQUIRE( value != 9);
// }


// BOOST_AUTO_TEST_CASE( LAYER_backward )
// {
//     Layer l1(4);
//     for (auto &n : l1._neurons) n._delta = 1;
//     Layer l2(2);
//     for (auto &n : l2._neurons) n._delta = 2;

//     for (auto &n1 : l1._neurons)
//         for (auto &n2 : l2._neurons)
//             connectNeurons(n1, n2, 1);

//     l1.backward();

//     for (auto &delta : l1.deltas())
//         BOOST_REQUIRE( delta != 1);
// }


// BOOST_AUTO_TEST_CASE( LAYER_modify )
// {
//     Layer l1 = {1, 1, 1, 1};
//     Layer l2 = {4};
//     for (auto &n : l2._neurons) n._delta = 1;

//     for (auto &n1 : l1._neurons)
//         for (auto &n2 : l2._neurons)
//             connectNeurons(n1, n2, 1);

//     // Learning rate vector
//     SnnValVec learningRate(l1._neurons.size()*l2._neurons.size());
//     fill(learningRate.begin(), learningRate.end(), 2);

//     l2.modify(learningRate.begin());

//     for (auto &n : l2._neurons)
//         for (auto &weight : n._inputWeights)
//             BOOST_REQUIRE( weight != 1);
// }

// BOOST_AUTO_TEST_SUITE_END()





// BOOST_AUTO_TEST_SUITE(PerceptronTest)

// BOOST_AUTO_TEST_CASE( PERCEPTRON_forward )
// {
//     Perceptron p = {2, 2, 2};
//     p.forward({1, 2});
// }


// BOOST_AUTO_TEST_CASE( PERCEPTRON_backward )
// {
//     Perceptron p = {2, 2, 2};
//     p.backward({1.4, 2});
// }


// BOOST_AUTO_TEST_CASE( PERCEPTRON_modify )
// {
//     Perceptron p = {2, 10, 2};
//     SnnValVec learningRate(p.getNumWeights());
//     fill(learningRate.begin(), learningRate.end(), 1);

//     p.forward({1.4, 2});
//     p.backward({1.4, 2});
//     p.modify(learningRate);
//     SnnValVec out1 =  p.getOutputs();

//     p.forward({1.4, 2});
//     p.backward({1.4, 2});
//     p.modify(learningRate);
//     SnnValVec out2 =  p.getOutputs();

//     BOOST_REQUIRE( out1 != out2);

// }


// BOOST_AUTO_TEST_CASE( PERCEPTRON_simple_backprop )
// {
//     SnnVal target = 0.415;

//     Perceptron p = {1, 1, 1, 1, 1, 1, 1};
//     SnnDataset trainSet = {
//         {1.0}, { target}
//     };

//     SnnValVec learningRate(p.getNumWeights());
//     fill(learningRate.begin(), learningRate.end(), 10);

//     train(p, trainSet, 10);

//     // BOOST_CHECK_CLOSE( p.getOutputs().front(), target, 0.1 );

//     // for (auto itSet =  trainSet.begin();
//     //         itSet != trainSet.end();
//     //         std::advance (itSet, 2)) {
//     //     SnnValVec &in = *itSet;
//     //     SnnValVec &out = *std::next(itSet);

//     //     p.forward(in);
//     //     cout << in << " " << p.getOutputs() << " " << out << std::endl;
//     //     cout.flush();
//     // }
// }

// BOOST_AUTO_TEST_CASE( PERCEPTRON_xor )
// {
//     Perceptron p = {2, 2, 1};

//     SnnDataset trainSet = {
//         {0, 0}, {0},
//         {1, 0}, {1},
//         {0, 1}, {1},
//         {1, 1}, {0}
//     };

//     cout << p.getNumWeights() << endl;
//     // cout.precision(4);
//     train(p, trainSet, 8000);

//     for (auto itSet =  trainSet.begin();
//             itSet != trainSet.end();
//             std::advance (itSet, 2)) {
//         SnnValVec &in = *itSet;
//         SnnValVec &out = *std::next(itSet);

//         p.forward(in);


//         cout << in << " " << p.getOutputs() << " " << out << std::endl;
//     }
// }

// BOOST_AUTO_TEST_SUITE_END()

