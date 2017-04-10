import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework




class mmod_loss(object):
    def __init__(self,
                 model,
                 input_size,
                 detector_size,
                 loss_per_false_alarm = 1,
                 loss_per_missed_target = 1,
                 truth_match_iou_threshold = 0.5,):

        self._model = model
        self._input_layer = model.input_layer
        self._input_size = input_size
        self._detector_size = detector_size
        self._loss_per_false_alarm = loss_per_false_alarm
        self._loss_per_missed_target = loss_per_missed_target
        self._truth_match_iou_threshold = truth_match_iou_threshold


    def __call__(self):
        pass


    def _gen_loss(self, samples, labels):
        pass


    def _collect_all_box_from_output(self, sess, input_samples, input_labels, output_samples, adjust_threshold):
        input_shape = input_samples.shape
        batch, row, col, chanel = output_samples.shape
        assert chanel == 1

        def _collect(b, r, c, d):
            score = output_samples[b, r, c, 0]
            if score <= adjust_threshold: return
            output_point = pt.create(r, c)
            pramid_point = self._model.map_output_input(sess, input_shape, output_point)
            rect = rt.centered_rect(pramid_point, self._detector_size)
            #self._input_layer.


        self.loop_all_images(batch, row, col, chanel, _collect)


        for b in xrange(batch):
            for r in xrange(row):
                for c in xrange(col):
                    pass




    def loop_all_images(self, batch, row, col, chanel, call_back):
        for b in xrange(batch):
            for r in xrange(row):
                for c in xrange(col):
                    for d in xrange(chanel):
                        call_back(b, r, c, d)