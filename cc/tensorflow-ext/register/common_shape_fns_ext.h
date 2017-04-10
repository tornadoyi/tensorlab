//
// Created by Jason on 10/04/2017.
//

#ifndef TENSORLAB_COMMON_SHAPE_FNS_EXT_H
#define TENSORLAB_COMMON_SHAPE_FNS_EXT_H


#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

namespace shape_inference_ext
{
#define DECL_UNCHANGED_INPUT_SHAPE(i) \
    Status UnchangedInput##i##Shape(shape_inference::InferenceContext* c);

    DECL_UNCHANGED_INPUT_SHAPE(0)
    DECL_UNCHANGED_INPUT_SHAPE(1)
    DECL_UNCHANGED_INPUT_SHAPE(2)
    DECL_UNCHANGED_INPUT_SHAPE(3)
    DECL_UNCHANGED_INPUT_SHAPE(4)
    DECL_UNCHANGED_INPUT_SHAPE(5)
}


#endif //PROJECT_COMMON_SHAPE_FNS_EXT_H
