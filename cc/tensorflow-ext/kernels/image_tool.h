//
// Created by Jason on 03/03/2017.
//

#ifndef TENSORLAB_IMAGE_TOOL_H
#define TENSORLAB_IMAGE_TOOL_H

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

#define DECLARE_IMAGE_TENSOR_DATA(name, data_revert_index) \
    inline int64 image_tensor_##name(const Tensor& tensor){ \
        auto dims = tensor.dims(); \
        auto index = dims - data_revert_index; \
        CHECK(dims >= 3); \
        CHECK(index >= 0); \
        return tensor.dim_size(index); \
    }

#define DECLARE_IMAGE_TTYPES_TENSOR_DATA(name, T, NDIMS,  data_revert_index) \
    inline int64 image_tensor_##name(TTypes<T, NDIMS>::Tensor tensor) { \
        auto dims = NDIMS; \
        auto index = dims - data_revert_index; \
        CHECK(dims >= 3); \
        CHECK(index >= 0); \
        return tensor.dimension(index); \
    }

#define DECLARE_IMAGE_TTYPES_CONST_TENSOR_DATA(name, T, NDIMS,  data_revert_index) \
    inline int64 image_tensor_##name(TTypes<T, NDIMS>::ConstTensor tensor) { \
        auto dims = NDIMS; \
        auto index = dims - data_revert_index; \
        CHECK(dims >= 3); \
        CHECK(index >= 0); \
        return tensor.dimension(index); \
    }


#define DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, TYPE, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_TENSOR_DATA(name, TYPE, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_CONST_TENSOR_DATA(name, TYPE, NDIMS, data_revert_index)



#define DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(name, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, int64, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, int32, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, uint16, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, int16, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, uint8, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, int8, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, Eigen::half, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, float, NDIMS, data_revert_index) \
    DECLARE_IMAGE_TTYPES_ALL_TENSOR(name, double, NDIMS, data_revert_index)



DECLARE_IMAGE_TENSOR_DATA(nbatch, 4)
DECLARE_IMAGE_TENSOR_DATA(nrow, 3)
DECLARE_IMAGE_TENSOR_DATA(ncol, 2)
DECLARE_IMAGE_TENSOR_DATA(ndata, 1)


DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(nbatch, 3, 4)
DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(nrow, 3, 3)
DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(ncol, 3, 2)
DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(ndata, 3, 1)


DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(nbatch, 4, 4)
DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(nrow, 4, 3)
DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(ncol, 4, 2)
DECLARE_IMAGE_TTYPES_TENSOR_DATA_ALL_TYPES(ndata, 4, 1)


#define _nb(t) image_tensor_nbatch(t)
#define _nr(t) image_tensor_nrow(t)
#define _nc(t) image_tensor_ncol(t)
#define _nd(t) image_tensor_ndata(t)











#endif //TENSORLAB_IMAGE_TOOL_H
