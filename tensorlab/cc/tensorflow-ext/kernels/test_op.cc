
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "image_tool.h"
#include "test_op.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Device, typename T>
class TestOp : public OpKernel
{
public:
    explicit TestOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& src_tensor = context->input(0);
        const Tensor& dst_tensor = context->input(1);


        OP_REQUIRES(context,
                    src_tensor.dims() == dst_tensor.dims(),
                    errors::InvalidArgument("image src and dst must be same ",
                                            "src: ", src_tensor.shape().DebugString(), " ",
                                            "dst: ", dst_tensor.shape().DebugString(), " "
                    ));



        Tensor* output1;
        OP_REQUIRES_OK(context, context->allocate_output(0, src_tensor.shape(), &output1));
        CHECK(output1->CopyFrom(src_tensor, src_tensor.shape()));

        Tensor* output2;
        OP_REQUIRES_OK(context, context->allocate_output(1, dst_tensor.shape(), &output2));
        CHECK(output2->CopyFrom(dst_tensor, dst_tensor.shape()));



    }

};



namespace kernel
{
    template <typename T>
    struct Test<CPUDevice, T>
    {
        void operator()(const CPUDevice& d,
                        typename TTypes<T, 4>::ConstTensor src_data,
                        typename TTypes<float, 2>::ConstTensor rects,
                        typename TTypes<int32, 2>::ConstTensor indexes,
                        typename TTypes<float, 4>::Tensor output_data) try
        {


        }
        catch(std::exception& e)
        {
            LOG(FATAL) << e.what();
        }
    };


#define DEFINE_CPU_SPECS(T)                     \
    template struct Test<CPUDevice, T>; \

    TF_CALL_REAL_NUMBER_TYPES(DEFINE_CPU_SPECS);

#undef DEFINE_CPU_SPECS
}


#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("Test")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          TestOp<CPUDevice, T>);



TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL




