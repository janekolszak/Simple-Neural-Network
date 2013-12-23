#ifndef SNN_TYPES_HPP
#define SNN_TYPES_HPP

#include <vector>
#include <list>
#include <iostream>
namespace snn
{

typedef double SnnVal;
typedef std::vector<SnnVal> SnnValVec;
typedef std::vector<std::reference_wrapper<SnnVal>> SnnValRefVec;
typedef std::list<SnnValVec> SnnDataset;

bool operator==(SnnValVec &valuesA, SnnValVec &valuesB) {
    if (valuesA.size() != valuesB.size())
        return false;
    return std::equal(valuesA.begin(), valuesA.end(), valuesB.begin());
}


bool operator!=(SnnValVec &valuesA, SnnValVec &valuesB) {
    return ! (valuesA == valuesB);
}


std::ostream &operator<<(std::ostream &os, const SnnValVec &values) {
    os <<"[";
    for (auto it = values.begin(); it != std::prev(values.end()); ++it)
        os << *it << " ";
    os << *std::prev(values.end());
    os <<"]";
    return os;
}

}

#endif