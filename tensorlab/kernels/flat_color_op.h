//
// Created by Jason on 08/03/2017.
//

#ifndef TENSORLAB_FLAT_COLOR_OP_H
#define TENSORLAB_FLAT_COLOR_OP_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace kernel
{

    template <typename Device, typename T, size_t NDIMS>
    struct FlatColor {

        void operator()(const Device& d,
                        typename TTypes<T, NDIMS>::ConstTensor input,
                        typename TTypes<T, NDIMS-1>::Tensor output);

    };

} // namespace kernel


#endif //TENSORLAB_FLAT_COLOR_OP_H
