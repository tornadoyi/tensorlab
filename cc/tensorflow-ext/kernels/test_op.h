//
// Created by Jason on 03/04/2017.
//

#ifndef TENSORLAB_TEST_OP_H
#define TENSORLAB_TEST_OP_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace kernel
{

    template <typename Device, typename T>
    struct Test {

        void operator()(const Device& d,
                        typename TTypes<T, 4>::ConstTensor src_data,
                        typename TTypes<float, 2>::ConstTensor rects,
                        typename TTypes<int32, 2>::ConstTensor indexes,
                        typename TTypes<float, 4>::Tensor output_data);

    };

} // namespace kernel


#endif //PROJECT_TEST_OP_H
