// #ifndef SNN_UTILS_HPP
// #define SNN_UTILS_HPP

// #include <iostream>
// #include <snn/basic/perceptron.hpp>

// using namespace std;
// namespace snn {



// // void train(Perceptron &perceptron, SnnDataset dataset, size_t numEpochs);
// void train(Perceptron &perceptron, SnnDataset dataset, size_t numEpochs) {


//     SnnValVec learningRate(perceptron.getNumWeights());
//     fill(learningRate.begin(), learningRate.end(), 0.03);

//     for (size_t i = 0; i < numEpochs; ++i) {
//         for (auto itSet =  dataset.begin();
//                 itSet != dataset.end();
//                 std::advance (itSet, 2)) {
//             SnnValVec &in = *itSet;
//             SnnValVec &out = *std::next(itSet);
//             // cout << "-----------------------------------" << endl;
//             // cout << "WAGI " <<   perceptron.getWeights() << endl;

//             perceptron.forward(in);
//             // cout << "OUT " << perceptron.getOutputs() << endl;
//             perceptron.backward(out);

//             perceptron.modify(learningRate);
//         }
//     }
// }

// } // namespace snn

// #endif