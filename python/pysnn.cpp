#include <boost/python.hpp>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/implicit.hpp>

#include <snn/snn.hpp>
#include <snn/types.hpp>
#include <vector>


BOOST_PYTHON_MODULE(pysnn)
{
    using namespace boost::python;
    using namespace snn;

    // Types
    class_<std::vector<double>>("SnnValVec")
        .def(vector_indexing_suite<std::vector<double>>())
    ;

    // Activation functions
    class_<LogSigmoid>("LogSigmoid", init<SnnVal, SnnVal>())
        .def("value", &LogSigmoid::value)
        .def("derivative", &LogSigmoid::derivative)
    ;

    class_<Linear>("Linear", init<SnnVal, SnnVal, SnnVal>())
        .def("value", &Linear::value)
        .def("derivative", &Linear::derivative)
    ;

    class_<LinearScalingFunction>("LinearScalingFunction", init<SnnVal, SnnVal, SnnVal, SnnVal>())
        .def("value", &LinearScalingFunction::value)
        .def("derivative", &LinearScalingFunction::derivative)
    ;



}