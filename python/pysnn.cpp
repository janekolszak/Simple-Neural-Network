#include <boost/python.hpp>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>


#include <snn/snn.hpp>
#include <snn/types.hpp>
#include <vector>


std::vector<double> qwerqwer( ) {
    std::vector<double> v;
    v.push_back(2.0);
    return v;
}

BOOST_PYTHON_MODULE(pysnn)
{
    using namespace boost::python;
    using namespace snn;

    // Types
    class_<std::vector<double>>("SnnValVec")
    .def(vector_indexing_suite<std::vector<double>>())
    ;

    def("qwerqwer", qwerqwer);



    // Activation functions
    def("logSigmoid", logSigmoid);
    def("logSigmoidDerivative", logSigmoidDerivative);

}