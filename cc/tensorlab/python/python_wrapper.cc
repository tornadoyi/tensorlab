#include <boost/python.hpp>

#include "bind_geometry.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(tensorlab)
{
    bind_geometry();

}

