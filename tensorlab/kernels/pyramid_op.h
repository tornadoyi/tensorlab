#ifndef TENSORLAB_KERNELS_PYRAMID_OP_H_
#define TENSORLAB_KERNELS_PYRAMID_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"


namespace kernel {

    template <typename Device, typename T>
    struct PyramidImages {
        void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor images,
                        const int left_image_count, int padding,
                        typename TTypes<float, 4>::Tensor pyramid);
    };

}  // namespace functor


#endif //TENSORLAB_KERNELS_PYRAMID_OP_H_