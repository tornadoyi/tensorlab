//
// Created by Jason on 06/03/2017.
//

#ifndef TENSORLAB_ASSIGN_IMAGE_OP_H
#define TENSORLAB_ASSIGN_IMAGE_OP_H

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace kernel
{

    template <typename Device, typename T, size_t NDIMS>
    struct AssignImage {

        void operator()(const Device& d,
                        typename TTypes<T, NDIMS>::ConstTensor src_data,
                        int startx,
                        int starty,
                        typename TTypes<T, NDIMS>::Tensor output_data);

    };

} // namespace kernel

#endif //TENSORLAB_ASSIGN_IMAGE_OP_H
