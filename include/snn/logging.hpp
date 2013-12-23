
#ifndef SNN_LOGGING_HPP
#define SNN_LOGGING_HPP

#include <iostream>
#include <vector> // TODO change to snn/types.hpp
#
namespace  snn {

// For now logging is just writing to stdout.
void LOGD(std::string message, std::vector<double>& data)
{
    std::cout << message;
    for (auto it = data.begin(); it != data.end(); ++it) {
        std::cout << *it << " ";
    }

    std::cout << std::endl;
}

void LOGD(std::string message)
{
    std::cout << message << std::endl;
}

}
#endif