//
// Created by Jason on 01/04/2017.
//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/resize_bilinear_op.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "image_tool.h"
#include "assign_image_op.h"
#include "pyramid_apply_op.h"


using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Device, typename T>
class PyramidApplyOp : public OpKernel
{
public:
    explicit PyramidApplyOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& input_tensor = context->input(0);
        const Tensor& size_tensor = context->input(1);
        const Tensor& rect_tensor = context->input(2);


        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::InvalidArgument("image must be 4-dimensional",
                                            input_tensor.shape().DebugString()));

        OP_REQUIRES(context, size_tensor.dims() == 1 && size_tensor.dim_size(0) == 2,
                    errors::InvalidArgument("size must be 1-dimensional, with (height, width)",
                                            input_tensor.shape().DebugString()));

        OP_REQUIRES(context, rect_tensor.dims() == 2 && rect_tensor.dim_size(1) == 4,
                    errors::InvalidArgument("rect shape must be N x 4 ",
                                            rect_tensor.shape().DebugString()));

        auto p_size = size_tensor.vec<int32>();
        auto output_height = p_size(0);
        auto output_width = p_size(1);


        Tensor* output_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                0, TensorShape({_nb(input_tensor), output_height, output_width, _nd(input_tensor)}),
                &output_tensor));


        Tensor indexes_tensor;
        OP_REQUIRES_OK(context, context->allocate_temp(
                DataTypeToEnum<int32>::v(),
                TensorShape({_nb(input_tensor) * rect_tensor.dim_size(0), 3}),
                &indexes_tensor));


        // generate indexes
        auto p_index = indexes_tensor.tensor<int32, 2>();
        auto batch = _nb(input_tensor);
        for(int64 i=0; i<_nb(input_tensor); ++i)
        {
            auto st = i * rect_tensor.dim_size(0);
            for(int32 j=0; j<rect_tensor.dim_size(0); ++j)
            {
                auto index = st + j;
                p_index(index, 0) = i;   // src
                p_index(index, 1) = i;   // dst
                p_index(index, 2) = j;   // rect

            }
        }


        kernel::AssignImage<Device, T>()(
                context->eigen_device<Device>(),
                input_tensor.tensor<T, 4>(),
                rect_tensor.tensor<float, 2>(),
                ((const Tensor)indexes_tensor).tensor<int32, 2>(),
                output_tensor->tensor<float, 4>()
        );


    }

};




#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("PyramidApply")        \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"), \
                          PyramidApplyOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL