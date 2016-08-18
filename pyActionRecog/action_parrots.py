import numpy as np
import sys

import cv2
from utils.io import flow_stack_oversample, rgb_to_parrots
import pyparrots.dnn as dnn


class ParrotsNet(object):

    def __init__(self, parrots_session_file, input_size=None):

        self._parrots_runner = dnn.Runner(parrots_session_file, extract=True)
        self._parrots_runner.setup()
        self._parrots_session = self._parrots_runner.session

        with self._parrots_session.flow('main') as flow:
            input_shape = flow.get_data_spec('data').shape[::-1]

        if input_size is not None:
            input_shape = input_shape[:2] + input_size

        self._sample_shape = input_shape
        self._channel_mean = [104, 117, 123]

    def predict_rgb_frame_list(self, frame_list,
                             score_name, over_sample=True,
                             multiscale=None, frame_size=None):

        if frame_size is not None:
            frame_list = [cv2.resize(x, frame_size) for x in frame_list]

        if over_sample:
            if multiscale is None:
                os_frame = np.concatenate([rgb_to_parrots(x, mean_val=self._channel_mean,
                                          crop_size=(self._sample_shape[2], self._sample_shape[3]))
                                     for x in frame_list], axis=0)
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame_list = [cv2.resize(x, (0, 0), fx=1.0 / scale, fy=1.0 / scale) for x in frame_list]
                    os_frame.extend(np.concatenate([rgb_to_parrots(x, mean_val=self._channel_mean,
                                          crop_size=(self._sample_shape[2], self._sample_shape[3]))
                                    for x in resized_frame_list]))
                os_frame = np.concatenate(os_frame, axis=0)
        else:
            os_frame = rgb_to_parrots(False)

        bs = self._sample_shape[0]

        feed_data = np.zeros(self._sample_shape)

        score_list = []
        for offset in xrange(0, os_frame.shape[0], bs):
            step = min(bs, os_frame.shape[0]-offset)
            feed_data[:step, ...] = os_frame[offset:offset+step, ...]

            with self._parrots_session.flow("main") as flow:
                flow.set_input('data', feed_data.astype(np.float32, order='C'))
                flow.forward()
                score_list.append(flow.data(score_name).value().T[:step])

        if over_sample:
            tmp = np.concatenate(score_list, axis=0)
            return tmp.reshape((len(os_frame) / 10, 10, score_list[0].shape[-1]))
        else:
            return np.concatenate(score_list, axis=0)

    def predict_flow_stack_list(self, flow_stack_list, score_name, over_sample=True, frame_size=None):

        if frame_size is not None:
            for i in xrange(len(flow_stack_list)):
                flow_stack_list[i] = np.array([cv2.resize(x, frame_size) for x in flow_stack_list[i]])

        if over_sample:
            tmp = [flow_stack_oversample(stack, (self._sample_shape[2], self._sample_shape[3]))
                                        for stack in flow_stack_list]
            os_frame = np.concatenate(tmp, axis=0)
        else:
            os_frame = np.array(flow_stack_list)

        os_frame -= 128

        bs = self._sample_shape[0]
        feed_data = np.zeros(self._sample_shape)

        score_list = []
        for offset in xrange(0, os_frame.shape[0], bs):
            step = min(bs, os_frame.shape[0] - offset)
            feed_data[:step, ...] = os_frame[offset:offset + step, ...]

            with self._parrots_session.flow("main") as flow:
                flow.set_input('data', feed_data.astype(np.float32, order='C'))
                flow.forward()
                score_list.append(flow.data(score_name).value().T[:step])

        if over_sample:
            tmp = np.concatenate(score_list, axis=0)
            return tmp.reshape((len(os_frame) / 10, 10, score_list[0].shape[-1]))
        else:
            return np.concatenate(score_list, axis=0)
