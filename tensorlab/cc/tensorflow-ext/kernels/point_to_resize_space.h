//
// Created by Jason on 01/04/2017.
//

#ifndef TENSORLAB_POINT_TO_RESIZE_SPACE_H
#define TENSORLAB_POINT_TO_RESIZE_SPACE_H


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace kernel
{

    template <typename Device, typename T>
    struct PointToResizeSpace {

        void operator()(const Device& d,
                        typename TTypes<T, 2>::ConstTensor points,
                        typename TTypes<float, 2>::ConstTensor scales,
                        typename TTypes<int32, 2>::ConstTensor indexes,
                        typename TTypes<T, 2>::Tensor output_data);

    };

} // namespace kernel


#endif //PROJECT_POINT_TO_RESIZE_SPACE_H
