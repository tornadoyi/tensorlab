//
// Created by Jason on 01/04/2017.
//

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "image_tool.h"
#include "point_to_resize_space.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Device, typename T>
class PointToResizeSpaceOp : public OpKernel
{
public:
    explicit PointToResizeSpaceOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& point_tensor = context->input(0);
        const Tensor& scale_tensor = context->input(1);
        const Tensor& indexes_tensor = context->input(2);


        OP_REQUIRES(context,
                    point_tensor.dims() == 2 && point_tensor.dim_size(1) == 2,
                    errors::InvalidArgument("point_tensor must be 2-dimensional, N x (y, x) ",
                                            point_tensor.shape().DebugString(), " "
                    ));


        OP_REQUIRES(context, scale_tensor.dims() == 2 &&  scale_tensor.dim_size(1) == 2,
                    errors::InvalidArgument("scale_tensor must be 2-dimensional with N x (scale_y, scale_x)",
                                            scale_tensor.shape().DebugString()));


        OP_REQUIRES(context, indexes_tensor.dims() == 2 && indexes_tensor.dim_size(1) == 2,
                    errors::InvalidArgument("rects_tensor must be 2-dimensional with (N x (point_index, scale_index))",
                                            indexes_tensor.shape().DebugString()));



        Tensor* output;
        OP_REQUIRES_OK(context, context->allocate_output(
                0,
                TensorShape({indexes_tensor.dim_size(0), 2}),
                &output));


        kernel::PointToResizeSpace<Device, T>()(context->eigen_device<Device>(),
                                                point_tensor.tensor<T, 2>(),
                                                scale_tensor.tensor<float, 2>(),
                                                indexes_tensor.tensor<int32, 2>(),
                                                output->tensor<float, 2>());

    }

};



namespace kernel
{
    template <typename T>
    struct PointToResizeSpace<CPUDevice, T>
    {
        void operator()(const CPUDevice& d,
                        typename TTypes<T, 2>::ConstTensor points,
                        typename TTypes<float, 2>::ConstTensor scales,
                        typename TTypes<int32, 2>::ConstTensor indexes,
                        typename TTypes<float, 2>::Tensor output_data)
        {
            auto n_points = points.dimension(0);
            auto n_scales = scales.dimension(0);

            auto index_count = indexes.dimension(0);

            for (int64 i = 0; i < index_count; ++i)
            {
                auto idx_p = indexes(i, 0);
                auto idx_s = indexes(i, 1);

                if(idx_p < 0 || idx_p >= n_points ||
                   idx_s < 0 || idx_s >= n_scales)
                {
                    LOG(ERROR) << "out of range "
                               << "idx_p: " << idx_p << "<" << n_points << " "
                               << "idx_s: " << idx_s << "<" << n_scales << " ";

                    continue;
                }

                auto y = points(idx_p, 0);
                auto x = points(idx_p, 1);
                auto s_y = scales(idx_s, 0);
                auto s_x = scales(idx_s, 1);


                auto in_y = (float)y * s_y;
                auto lower_in_y = static_cast<int64>(in_y);
                auto upper_in_y = lower_in_y + 1;
                auto lerp_in_y = in_y - lower_in_y;
                auto dst_y = (int64)((upper_in_y - lower_in_y) * lerp_in_y);

                auto in_x = (float)x * s_x;
                auto lower_in_x = static_cast<int64>(in_x);
                auto upper_in_x = lower_in_x + 1;
                auto lerp_in_x = in_x - lower_in_x;
                auto dst_x = (int64)((upper_in_x - lower_in_x) * lerp_in_x);

                output_data(i, 0) = dst_y;
                output_data(i, 1) = dst_x;
            }
        }

    };


#define DEFINE_CPU_SPECS(T)                     \
    template struct PointToResizeSpace<CPUDevice, T>; \

    TF_CALL_REAL_NUMBER_TYPES(DEFINE_CPU_SPECS);

#undef DEFINE_CPU_SPECS
}


#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("PointToResizeSpace")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          PointToResizeSpaceOp<CPUDevice, T>);



TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL