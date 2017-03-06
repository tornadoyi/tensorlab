
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

        OP_REQUIRES(context, src_tensor.dims() == 3 && dst_tensor.dims() == 3,
                    errors::InvalidArgument("image must be 3-dimensional",
                                            src_tensor.shape().DebugString(), dst_tensor.shape().DebugString()));

        OP_REQUIRES(context, start_loc.dims() == 1,
                    errors::InvalidArgument("location must be 1-dimensional",
                                            start_loc.shape().DebugString()));

        auto Svec = start_loc.vec<int32>();
        int startx = Svec(0);
        int starty = Svec(1);

        Tensor* output;
        OP_REQUIRES_OK(context, context->allocate_output(0, dst_tensor.shape(), &output));
        CHECK(output->CopyFrom(dst_tensor, dst_tensor.shape()));

        typename TTypes<T, 3>::ConstTensor src_data = src_tensor.tensor<T, 3>();
        typename TTypes<T, 3>::Tensor output_data = output->tensor<T, 3>();


        kernel::AssignImage<Device, T> m;
        m(context->eigen_device<Device>(), src_data, output_data, startx, starty);

    }
};



namespace kernel
{
    template <typename T>
    struct AssignImage<CPUDevice, T>
    {
        void operator()(const CPUDevice& d,
                        typename TTypes<T, 3>::ConstTensor src,
                        typename TTypes<T, 3>::Tensor dst,
                        int startx, int starty)
        {
            /*
            LOG(INFO)<< _nr(src) << " " << _nc(src);
            LOG(INFO)<< _nr(dst) << " " << _nc(dst);
            LOG(INFO)<< startx << " " << starty;
            */

            auto channel = _nd(src);
            const T* ptr_src = src.data();
            T* ptr_dst = dst.data();


            auto r_count = _nc(src) * channel;
            auto src_start = 0;
            auto dst_start = (startx * _nc(src) + starty) * channel;

            for(int64 i=0; i<_nr(src); ++i)
            {
                auto p_src = src_start;
                auto p_dst = dst_start;
                for(int64 j=0; j<_nc(src); ++j)
                {
                    p_src += channel;
                    p_dst += channel;
                    ptr_dst[p_dst + 0] = ptr_src[p_src + 0];
                    ptr_dst[p_dst + 1] = ptr_src[p_src + 1];
                    ptr_dst[p_dst + 2] = ptr_src[p_src + 2];
                }
                src_start += r_count;
                dst_start += r_count;
            }
        }

    };
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
