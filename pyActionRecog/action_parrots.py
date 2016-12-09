import numpy as np
import sys

import cv2
from utils.io import flow_stack_oversample, rgb_to_parrots, fast_list2arr
import pyparrots.dnn as dnn
import pyparrots.base as base
import yaml


MAX_BATCHSIZE = 32
base.set_debug_log(False)


class ParrotsNet(object):

    def __init__(self, model_spec, weights,
                 device_id=None, batch_size=None,
                 flow_name='main', session_tmpl='data/parrots_session.tmpl',
                 score_name='prob', input_name='data', mem_opt=True):
        session_tmpl = yaml.load(open(session_tmpl))

        if device_id is not None:
            session_tmpl['flows'][0][flow_name]['devices'] = 'gpu({})'.format(device_id)

        if batch_size is not None:
            session_tmpl['flows'][0][flow_name]['batch_size'] = batch_size \
                if batch_size < MAX_BATCHSIZE else MAX_BATCHSIZE
        else:
            session_tmpl['flows'][0][flow_name]['batch_size'] = MAX_BATCHSIZE

        self._parrots_batch_size = session_tmpl['flows'][0][flow_name]['batch_size']

        session_tmpl['flows'][0][flow_name]['spec']['inputs'] = [input_name]
        session_tmpl['flows'][0][flow_name]['spec']['outputs'] = [score_name]

        self._input_name = input_name
        self._score_name = score_name
        self._flow_name = flow_name

        session_tmpl['use_dynamic_memory'] = mem_opt

        ready_session_str = yaml.dump(session_tmpl)

        # create model
        self._parrots_model = dnn.Model.from_yaml_text(open(model_spec).read())

        self._parrots_session = dnn.Session.from_yaml_text(self._parrots_model, ready_session_str)
        self._parrots_session.setup()

        # load trained model weights
        self._parrots_session.flow(self._flow_name).load_param(weights)

        with self._parrots_session.flow(self._flow_name) as flow:
            input_shape = flow.get_data_spec(self._input_name).shape[::-1]

        self._sample_shape = input_shape
        self._channel_mean = [104, 117, 123]

        self._parrots_flow = self._parrots_session.flow("main")

        self._subtract_buffer = np.ones(self._sample_shape, dtype=np.float32) * 128.0

    def predict_single_rgb_frame(self, frame, over_sample=True,
                             multiscale=None, frame_size=None):

        if frame_size is not None:
            frame = cv2.resize(frame, frame_size)

        if over_sample:
            if multiscale is None:
                os_frame = rgb_to_parrots(frame, mean_val=self._channel_mean,
                                          crop_size=(self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = cv2.resize(frame, (0, 0), fx=1.0 / scale, fy=1.0 / scale)
                    os_frame.extend(np.concatenate(rgb_to_parrots(resized_frame, mean_val=self._channel_mean,
                                          crop_size=(self._sample_shape[2], self._sample_shape[3]))))
                os_frame = np.concatenate(os_frame, axis=0)
        else:
            os_frame = rgb_to_parrots(frame, False)

        bs = self._sample_shape[0]

        feed_data = np.zeros(self._sample_shape)

        score_list = []
        for offset in xrange(0, os_frame.shape[0], bs):
            step = min(bs, os_frame.shape[0]-offset)
            feed_data[:step, ...] = os_frame[offset:offset+step, ...]

            self._parrots_flow.set_input('data', feed_data.T)
            self._parrots_flow.forward()
            score_list.append(self._parrots_flow.data(self._score_name).value().T[:step])

        return np.concatenate(score_list, axis=0)

    def predict_single_flow_stack(self, flow_stack, over_sample=True, frame_size=None):
        if frame_size is not None:
            flow_stack = fast_list2arr([cv2.resize(x, frame_size) for x in flow_stack], dtype=np.float32)

        if over_sample:
            os_frame = flow_stack_oversample(flow_stack, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([flow_stack])

        os_frame = os_frame - self._subtract_buffer
        
 	#print os_frame[0]

        self._parrots_flow.set_input('data', os_frame.T)

        self._parrots_flow.forward()
        score = self._parrots_flow.data(self._score_name).value().T.copy()
        #print score[0,:]

        return score
