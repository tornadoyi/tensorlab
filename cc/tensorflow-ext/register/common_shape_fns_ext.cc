//
// Created by Jason on 10/04/2017.
//

#include "common_shape_fns_ext.h"

using namespace tensorflow;

namespace shape_inference_ext
{
#define IMPL_UNCHANGED_INPUT_SHAPE(i) \
    Status UnchangedInput##i##Shape(shape_inference::InferenceContext* c) { \
        c->set_output(0, c->input(i)); \
        return Status::OK(); \
    }

    IMPL_UNCHANGED_INPUT_SHAPE(0)
    IMPL_UNCHANGED_INPUT_SHAPE(1)
    IMPL_UNCHANGED_INPUT_SHAPE(2)
    IMPL_UNCHANGED_INPUT_SHAPE(3)
    IMPL_UNCHANGED_INPUT_SHAPE(4)
    IMPL_UNCHANGED_INPUT_SHAPE(5)

}