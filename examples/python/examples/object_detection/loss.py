import numpy as np
import tensorflow as tf
import tensorlab as tl
from tensorlab import framework
from tensorlab.python.ops.geometry import rectangle_yx as rt, point_yx as pt




class mmod_loss(object):
    def __init__(self,
                 model,
                 detector_size,
                 loss_per_false_alarm = 1,
                 loss_per_missed_target = 1,
                 truth_match_iou_threshold = 0.5,):

        self._model = model
        self._input_layer = model.input_layer
        self._detector_size = np.array(detector_size, np.int32)
        self._loss_per_false_alarm = loss_per_false_alarm
        self._loss_per_missed_target = loss_per_missed_target
        self._truth_match_iou_threshold = truth_match_iou_threshold

        self._gen_loss()


    @property
    def loss_tensor(self): return self._loss_tensor


    def gen_input_dict(self, images, rects, groups):
        return {
            self._input_layer.input_images: images,
            self._input_rect_tensor: rects,
            self._input_groups_tensor: groups
        }


    def _gen_loss(self):
        # input
        self._input_rect_tensor = tf.placeholder(tf.int32, (None, 4))
        self._input_groups_tensor = tf.placeholder(tf.int32, (None, ))
        input_groups = tf.reshape(self._input_groups_tensor, (-1, 1))

        # create gradient and scale
        b, r, c, d = tl.dims(self._model.out)
        score_images = tf.reshape(self._model.out, (b, r, c))
        score_images_shape = tf.shape(score_images)
        grad = tf.zeros_like(score_images)
        scale = 1.0 #/ tl.len(score_images)


        # map rects to score points
        truth_score_points = self.image_rect_to_feat_coord(self._input_rect_tensor)
        truth_score_loc = tf.concat([input_groups, truth_score_points], 1)


        # predict all possible rects
        pred_scores, pred_rects, pred_output_points, pred_groups = self.collect_valid_rects(score_images, -self._loss_per_false_alarm)


        # The loss will measure the number of incorrect detections.  A detection is
        # incorrect if it doesn't hit a truth rectangle or if it is a duplicate detection
        # on a truth rectangle.
        grad -= tf.sparse_to_dense(truth_score_loc, score_images_shape, tf.tile([scale], [tl.len(truth_score_loc)]), validate_indices=False)


        def loop_images(s, grad):
            b = s.step
            image = score_images[b]
            truth = tf.gather_nd(self._input_rect_tensor, tf.where(tf.equal(self._input_groups_tensor, b)))

            # get dets, scroes and points at current image
            pred_index = tf.where(tf.equal(pred_groups, b))
            dets = tf.gather_nd(pred_rects, pred_index)
            scores = tf.gather_nd(pred_scores, pred_index)
            points = tf.gather_nd(pred_output_points, pred_index)

            # use part of dets, scores and points
            max_num_dets = 50 + tl.len(truth) * 5
            num_dets = tf.minimum(max_num_dets, tl.len(dets))
            dets, scores, points = dets[0:num_dets], scores[0:num_dets], points[0:num_dets]

            #dets = tl.Print(dets, [tl.len(dets)], message="dets")

            # The point of this loop is to fill out the truth_score_hits array.
            def gen_truth_score_hits(s, *args):
                i = s.step
                rect = dets[i]

                def do(truth_score_hits, hit_truth_table, final_dets):
                    # add rect to final_dets
                    final_dets = tf.concat([final_dets, tf.expand_dims(rect, 0)], 0)

                    # find best matched truth
                    match_index, match_score = self.find_best_match(rect, truth)

                    # update truth_score_hits
                    def update(truth_score_hits, hit_truth_table):

                        score = tf.cond(tf.greater(hit_truth_table[match_index], 0),
                                        lambda : scores[i] + self._loss_per_false_alarm,
                                        lambda : scores[i])

                        hit_truth_table += tf.sparse_to_dense([match_index], tf.shape(hit_truth_table), 1)
                        truth_score_hits += tf.sparse_to_dense([match_index], tf.shape(truth_score_hits), score)

                        return truth_score_hits, hit_truth_table

                    truth_score_hits, hit_truth_table = tf.cond(
                        tf.greater(match_score, self._truth_match_iou_threshold),
                        lambda: update(truth_score_hits, hit_truth_table),
                        lambda : (truth_score_hits, hit_truth_table))

                    return truth_score_hits, hit_truth_table, final_dets

                return tf.cond(self.overlaps_any_box_nms(rect, final_dets, iou_thresh=0.4), lambda: args, lambda: do(*args))

            truth_score_hits = tf.fill([tl.len(truth)], tf.to_float(0))
            hit_truth_table = tf.fill([tl.len(truth)], tf.to_int32(0))
            final_dets = tf.constant(0, tf.int32, (0, 4))
            truth_score_hits, hit_truth_table, final_dets = tl.for_loop(gen_truth_score_hits, 0, tl.len(dets),
                                                                        loop_vars=[truth_score_hits, hit_truth_table, final_dets],
                                                                        auto_var_shape=True)


            # Now figure out which detections jointly maximize the loss and detection score sum.  We
            # need to take into account the fact that allowing a true detection in the output, while
            # initially reducing the loss, may allow us to increase the loss later with many duplicate
            # detections.

            def gen_final_dets(s, *args):
                i = s.step
                rect = dets[i]

                def do(*args):

                    # find best matched truth
                    match_index, match_score = self.find_best_match(rect, truth)

                    def update(final_dets, final_det_indexes):
                        # add rect to final_dets
                        final_dets = tf.concat([final_dets, tf.expand_dims(rect, 0)], 0)
                        final_det_indexes = tf.concat([final_det_indexes, [i]], 0)
                        return final_dets, final_det_indexes


                    # 1. if match_index < 0 then truth size is 0, false alarm need punishment
                    # 2. if match_score > truth_match_iou_threshold
                    #       - && truth_score_hits[match_index] <= loss_per_missed_target
                    #            means that hit truth but not enough score for it, so that consider rect as missing
                    #            rect int this condition
                    #       - && truth_score_hits[match_index] > loss_per_missed_target
                    #            means that hit truth
                    # 3. else not hit any truth, false alarm need punishment
                    return tf.cond(tf.less(match_index, 0),
                                   lambda: update(*args),
                                   lambda: tf.cond(tl.logical_and(
                                       tf.greater(match_score, self._truth_match_iou_threshold),
                                       tf.less_equal(truth_score_hits[match_index], self._loss_per_missed_target)),
                                       lambda : args,
                                       lambda : update(*args))
                                   )


                return tf.cond(self.overlaps_any_box_nms(rect, final_dets, iou_thresh=0.4), lambda: args, lambda: do(*args))


            final_dets = tf.constant(0, tf.int32, (0, 4))
            final_det_indexes = tf.constant(0, tf.int32, (0, ))
            final_dets, final_det_indexes = tl.for_loop(gen_final_dets, 0, tl.len(dets),
                                                      loop_vars=[final_dets, final_det_indexes],
                                                        auto_var_shape=True)


            # update grad
            #final_det_indexes = tl.Print(final_det_indexes, [tl.len(final_det_indexes), tf.shape(final_det_indexes)], message="final_det_indexes ")

            final_points = tf.gather(points, final_det_indexes)
            vec_b = tf.tile([[b]], [tl.len(final_points), 1])
            indexes = tf.concat([vec_b, final_points], 1)
            grad += tf.sparse_to_dense(indexes, score_images_shape, tf.tile([scale], [tl.len(indexes)]), validate_indices=False)
            return grad


        grad = tl.for_loop(loop_images, 0, b, loop_vars=[grad])

        y_images = tf.stop_gradient(grad + score_images)
        self._loss_tensor = tf.nn.l2_loss(y_images - score_images) / tf.to_float(tf.reduce_prod(score_images_shape))





    def collect_valid_rects(self, images, adjust_threshold):
        # find all points > adjust_threshold
        batch_points = tf.where(tf.greater(images, adjust_threshold))

        def collect(batch_points):
            batch_points = tf.to_int32(batch_points)
            scores = tf.gather_nd(images, batch_points)
            groups, points = tf.split(batch_points, [1, -1], axis=1)
            groups, points = tl.flatten(groups), tf.reshape(points, (-1, 2))

            # map points from CNN space to pyramid space
            pyramid_points = self._model.gen_map_output_to_input_tensor(points)
            pyramid_rects = rt.centered_rect(pyramid_points, self._detector_size)

            # map pyramid rects to original image space
            rects = self._input_layer.gen_rect_from_output_space_to_input_space(pyramid_rects)

            # sort by scores
            scores, indexes = tf.nn.top_k(scores, tl.len(scores))
            rects = tf.gather(rects, indexes)
            points = tf.gather(points, indexes)
            groups = tf.gather(groups, indexes)

            return scores, rects, points, groups

        def empty():
            return tf.constant(0, tf.float32, [0,]), \
                   tf.constant(0, tf.int32, [0, 4]), \
                   tf.constant(0, tf.int32, [0, 2]), \
                   tf.constant(0, tf.int32, [0,])

        return tf.cond(tf.equal(tl.len(batch_points), 0), lambda: empty(), lambda: collect(batch_points))



    def image_rect_to_feat_coord(self, rects):
        def transform_rects(rects):
            # scale
            scales = tf.to_float(self._detector_size) / tf.to_float(rt.size(rects))
            scales = tf.maximum(scales[:, 0], scales[:, 1])
            scales = tf.minimum(scales, 1.0)

            # map rect to pyramid space
            pyramid_rects = self._input_layer.gen_rect_from_input_space_to_output_space(rects, scales)
            centers = rt.center(pyramid_rects)

            # map point from pyramid to CNN space
            return self._model.gen_map_input_to_output_tensor(centers)

        return tf.cond(tf.equal(tl.len(rects), 0),
                       lambda: tf.constant(0, rects.dtype, shape=(0, 2)),
                       lambda: transform_rects(rects))



    def overlaps_any_box_nms(self, rect, rects, iou_thresh=0.5, percent_covered_thresh=1.0):
        def check_overlap(rect, rects):
            rect, rects = tf.to_float(rect), tf.to_float(rects)
            inners, outers, iou_s = self._match_rects(rect, rects)
            index = tf.to_int32(tf.argmax(iou_s))

            iou = iou_s[index]
            inner = inners[index]
            max_rect = rects[index]
            overlap = tl.logical_or(
                tf.greater(iou, iou_thresh),
                tf.greater(inner / rt.area(rect), percent_covered_thresh),
                tf.greater(inner / rt.area(max_rect), percent_covered_thresh)
            )
            return overlap

        return tf.cond(tf.equal(tl.len(rects), 0), lambda: tf.constant(False), lambda: check_overlap(rect, rects))



    def find_best_match(self, rect, rects):

        def match(rect, rects):
            rect, rects = tf.to_float(rect), tf.to_float(rects)
            inners, outers, iou_s = self._match_rects(rect, rects)
            max_iou_idx = tf.to_int32(tf.argmax(iou_s))
            iou = iou_s[max_iou_idx]

            return max_iou_idx, iou

        return tf.cond(tf.equal(tl.len(rects), 0),
                       lambda:(tf.constant(-1, tf.int32), tf.constant(0, tf.float32)),
                       lambda: match(rect, rects))



    def _match_rects(self, rect, rects):
        test_rects = tf.tile(tf.expand_dims(rect, 0), [tl.len(rects), 1])

        inners = rt.area(rt.intersect(test_rects, rects))
        outers = rt.area(rt.union(test_rects, rects))
        outers = tf.maximum(outers, tl.epsilon)
        iou_s = tf.minimum(inners / outers, 1.0)

        return inners, outers, iou_s



    def debug_test(self, sess, crop_images, crop_rects, rect_groups):

        rects = tf.convert_to_tensor([[50, 50, 109, 109]], tf.int32)
        map_rects = self.image_rect_to_feat_coord(rects)

        r = sess.run(map_rects, self.gen_input_dict(crop_images, crop_rects, rect_groups))

        print(r)