from ..tensor import Tensor
import tensorflow as tf
import numpy as np
from .. import utils


# reference by tensorflow/core/framework/common_shape_fns.cc
# switch (padding_type) {
#   case Padding::VALID:
#       *output_size = (input_size - filter_size + stride) / stride;
#       *padding_before = *padding_after = 0;
#   case Padding::SAME:
#       *output_size = (input_size + stride - 1) / stride;
#       const int64 padding_needed = std::max(0LL, (*output_size - 1) * stride + filter_size - input_size);
#       *padding_before = padding_needed / 2;
#       *padding_after = padding_needed - *padding_before;


class conv2d(Tensor):
    def __init__(self, *args, **kwargs):
        Tensor.__init__(self, tf.nn.conv2d, *args, **kwargs)

    # input_shape: 1-D of length 2 vector, HW format
    def padding_shape(self, input_shape):
        filter = self.arg_attr('filter')
        strides = self.arg_attr('strides')
        padding = self.arg_attr('padding')

        if padding == 'VALID':
            return (0, 0)

        else:
            output_height = (input_shape[0] + strides[1] - 1) / strides[1]
            output_width = (input_shape[1] + strides[2] - 1) / strides[2]

            padding_needed_height = np.max((0, (output_height - 1) * strides[1] + filter[0] - input_shape[0]))
            padding_needed_width = np.max((0, (output_width - 1) * strides[2] + filter[1] - input_shape[1]))

            return (padding_needed_height/2, padding_needed_width/2)



    # input_shape: 1-D of length 4 vector, NHWC format
    def output_shape(self, input_shape):
        assert np.shape(input_shape) == (4,)

        filter = utils.shape(self.arg_attr('filter'))
        strides = self.arg_attr('strides')
        padding = self.arg_attr('padding')

        if padding == 'VALID':
            output_height = (input_shape[1] - filter[0] + strides[1]) / strides[1]
            output_width = (input_shape[2] - filter[1] + strides[2]) / strides[2]

        else:
            output_height = (input_shape[1] + strides[1] - 1) / strides[1]
            output_width = (input_shape[2] + strides[2] - 1) / strides[2]

        return (input_shape[0], output_height, output_width, filter[3])


