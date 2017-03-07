
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

        OP_REQUIRES(context, src_tensor.dims() == 3 && dst_tensor.dims() == 3,
                    errors::InvalidArgument("image must be 3-dimensional",
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

        typename TTypes<T, 3>::ConstTensor src_data = src_tensor.tensor<T, 3>();
        typename TTypes<T, 3>::Tensor output_data = output->tensor<T, 3>();


        kernel::AssignImage<Device, T>()(context->eigen_device<Device>(), src_data, output_data, startx, starty);

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

            //LOG(INFO)<< "src: "<< _nr(src) << " " << _nc(src);
            //LOG(INFO)<< "dst: "<< _nr(dst) << " " << _nc(dst);
            //LOG(INFO)<< startx << " " << starty;


            auto channel = _nd(src);
            const T* ptr_src = src.data();
            T* ptr_dst = dst.data();

            auto count_src_row = _nc(src) * channel;
            auto count_dst_row = _nc(dst) * channel;
            int64 st_src = 0;
            int64 st_dst = (startx * _nc(dst) + starty) * channel;
            for(int64 i=0; i<_nr(src); ++i)
            {
                auto p_src = st_src;
                auto p_dst = st_dst;
                for (int64 j = 0; j < _nc(src); ++j)
                {
                    //auto p_src = (i * _nc(src) + j) * channel;
                    //auto p_dst = ((i+startx) * _nc(dst) + (j+starty)) * channel;
                    ptr_dst[p_dst + 0] = ptr_src[p_src + 0];
                    ptr_dst[p_dst + 1] = ptr_src[p_src + 1];
                    ptr_dst[p_dst + 2] = ptr_src[p_src + 2];

                    p_src += channel;
                    p_dst += channel;
                }
                st_src += count_src_row;
                st_dst += count_dst_row;
            }

        }

    };


#define DEFINE_CPU_SPECS(T)                     \
  template struct AssignImage<CPUDevice, T>;

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
