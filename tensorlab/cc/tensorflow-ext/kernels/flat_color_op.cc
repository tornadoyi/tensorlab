
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "image_tool.h"
#include "flat_color_op.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Device, typename T>
class FlatColorOp : public OpKernel
{
public:
    explicit FlatColorOp(OpKernelConstruction* context) : OpKernel(context){}

    void Compute(OpKernelContext* context) override
    {
        const Tensor& input_tensor = context->input(0);

        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::InvalidArgument("image must be 3-dimensional or 4-dimensional",
                                            input_tensor.shape().DebugString()));


        Tensor* output;
        auto output_shape = TensorShape({_nb(input_tensor), _nr(input_tensor) * _nd(input_tensor), _nc(input_tensor)});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));


        kernel::FlatColor<Device, T, 4>()(context->eigen_device<Device>(),
                                          input_tensor.tensor<T, 4>(),
                                          output->tensor<T, 3>());

    }
};



namespace kernel
{
    template <typename T, size_t NDIMS>
    struct FlatColor<CPUDevice, T, NDIMS>
    {
        void operator()(const CPUDevice& d,
                        typename TTypes<T, NDIMS>::ConstTensor src,
                        typename TTypes<T, NDIMS-1>::Tensor dst)
        {
            int64 batch = NDIMS <= 3 ? 1 : _nb(src);

            auto channel = _nd(src);
            const T* ptr_src = src.data();
            T* ptr_dst = dst.data();

            auto size_img = _nr(src) * _nc(src) * channel;
            auto offset = _nr(src) * _nc(src);

            for(int64 n=0; n<batch; ++n)
            {
                int64 p_src = 0;
                int64 p_dst = 0;
                for(int64 i=0; i<_nr(src); ++i)
                {
                    for (int64 j = 0; j < _nc(src); ++j)
                    {
                        const T* ptr_cur_src = &ptr_src[p_src];
                        T* ptr_cur_dst = &ptr_dst[p_dst];

                        auto offset_dst = 0;
                        for(int64 k=0; k<channel; ++k)
                        {
                            ptr_cur_dst[offset_dst] = ptr_cur_src[k];
                            offset_dst += offset;
                        }

                        p_src += channel;
                        p_dst += 1;
                    }
                }

                ptr_src += size_img;
                ptr_dst += size_img;
            }
        }

    };


#define DEFINE_CPU_SPECS(T)                     \
    template struct FlatColor<CPUDevice, T, 3>; \
    template struct FlatColor<CPUDevice, T, 4>;

    TF_CALL_REAL_NUMBER_TYPES(DEFINE_CPU_SPECS);

#undef DEFINE_CPU_SPECS
}


#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("FlatColor")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          FlatColorOp<CPUDevice, T>);



TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL



