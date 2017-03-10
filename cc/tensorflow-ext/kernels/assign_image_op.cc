
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "image_tool.h"
#include "assign_image_op.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Device, typename T>
class AssignImageOp : public OpKernel
{
public:
    explicit AssignImageOp(OpKernelConstruction* context) : OpKernel(context){}

    void Compute(OpKernelContext* context) override
    {
        const Tensor& src_tensor = context->input(0);
        const Tensor& dst_tensor = context->input(1);
        auto start_loc = context->input(2);
        auto Svec = start_loc.vec<int32>();
        int startx = Svec(0);
        int starty = Svec(1);

        OP_REQUIRES(context, src_tensor.dims() == 3 || src_tensor.dims() == 4,
                    errors::InvalidArgument("image must be 3-dimensional or 4-dimensional",
                                            src_tensor.shape().DebugString(), dst_tensor.shape().DebugString()));

        OP_REQUIRES(context, src_tensor.dims() == dst_tensor.dims(),
                    errors::InvalidArgument("src and dst must be same dimensional",
                                            src_tensor.shape().DebugString(), dst_tensor.shape().DebugString()));

        OP_REQUIRES(context, start_loc.dims() == 1,
                    errors::InvalidArgument("location must be 1-dimensional",
                                            start_loc.shape().DebugString()));

        OP_REQUIRES(context, startx + _nr(src_tensor) <= _nr(dst_tensor) &&
                starty + _nc(src_tensor) <= _nc(dst_tensor),
                    errors::InvalidArgument("assign image out of bound",
                                            start_loc.shape().DebugString()));

        Tensor* output;
        OP_REQUIRES_OK(context, context->allocate_output(0, dst_tensor.shape(), &output));
        CHECK(output->CopyFrom(dst_tensor, dst_tensor.shape()));


        if(src_tensor.dims() == 3)
        {
            kernel::AssignImage<Device, T, 3>()(context->eigen_device<Device>(),
                                                src_tensor.tensor<T, 3>(),
                                                startx, starty,
                                                output->tensor<T, 3>());
        }
        else
        {
            kernel::AssignImage<Device, T, 4>()(context->eigen_device<Device>(),
                                                src_tensor.tensor<T, 4>(),
                                                startx, starty,
                                                output->tensor<T, 4>());
        }

    }
};



namespace kernel
{
    template <typename T, size_t NDIMS>
    struct AssignImage<CPUDevice, T, NDIMS>
    {
        void operator()(const CPUDevice& d,
                        typename TTypes<T, NDIMS>::ConstTensor src,
                        int startx,
                        int starty,
                        typename TTypes<T, NDIMS>::Tensor dst)
        {
            int64 batch = NDIMS <= 3 ? 1 : _nb(src);

            auto channel = _nd(src);
            const T* ptr_src = src.data();
            T* ptr_dst = dst.data();

            auto size_src_row = _nc(src) * channel;
            auto size_dst_row = _nc(dst) * channel;
            auto size_src_img = size_src_row * _nr(src);
            auto size_dst_img = size_dst_row * _nr(dst);

            int64 init_src = 0;
            int64 init_dst = (startx * _nc(dst) + starty) * channel;
            for(int64 n=0; n<batch; ++n)
            {
                auto st_src = init_src;
                auto st_dst = init_dst;
                for(int64 i=0; i<_nr(src); ++i)
                {
                    auto p_src = st_src;
                    auto p_dst = st_dst;
                    for (int64 j = 0; j < _nc(src); ++j)
                    {
                        for(int64 k=0; k<channel; ++k)
                            ptr_dst[p_dst + k] = ptr_src[p_src + k];

                        p_src += channel;
                        p_dst += channel;
                    }
                    st_src += size_src_row;
                    st_dst += size_dst_row;
                }

                ptr_src += size_src_img;
                ptr_dst += size_dst_img;
            }
        }

    };


#define DEFINE_CPU_SPECS(T)                     \
    template struct AssignImage<CPUDevice, T, 3>; \
    template struct AssignImage<CPUDevice, T, 4>;

    TF_CALL_REAL_NUMBER_TYPES(DEFINE_CPU_SPECS);

#undef DEFINE_CPU_SPECS
}


#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("AssignImage")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          AssignImageOp<CPUDevice, T>);



TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL



REGISTER_OP("AssignImage")
        .Input("src: T")
        .Input("dst: T")
        .Input("location: int32")
        .Output("output_image: T")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");
