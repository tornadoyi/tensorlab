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

    template <typename Device, typename T>
    struct AssignImage {
        void operator()(const Device& d,
                        typename TTypes<T, 3>::ConstTensor src_data,
                        typename TTypes<T, 3>::Tensor output_data,
                        int startx,
                        int starty);
    };

} // namespace kernel

#endif //TENSORLAB_ASSIGN_IMAGE_OP_H
