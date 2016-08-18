import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from pyActionRecog import parse_directory
from pyActionRecog import parse_split_file

from pyActionRecog.utils.video_funcs import default_aggregation_func

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('split', type=int, choices=[1, 2, 3],
                    help='on which split to test the network')
parser.add_argument('modality', type=str, choices=['rgb', 'flow'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('parrots_session', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument('--num_frame_per_video', type=int, default=25,
                    help="prefix of y direction flow images")
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_worker', type=int, default=1)
parser.add_argument("--parrots_path", type=str, default='/home/yjxiong/Parrots/pyparrots',
                    help='path to the Parrots toolbox')
args = parser.parse_args()

print args

sys.path.append(args.parrots_path)
from pyActionRecog.action_parrots import ParrotsNet


def build_net():
    global net
    net = ParrotsNet(args.parrots_session)
build_net()

# build neccessary information
print args.dataset
split_tp = parse_split_file(args.dataset)
f_info = parse_directory(args.frame_path,
                         args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix)


eval_video_list = split_tp[args.split - 1][1]

score_name = 'fc_action'


def eval_video(video):
    global net
    label = video[1]
    vid = video[0]

    video_frame_path = f_info[0][vid]
    frame_cnt = f_info[1][vid]

    stack_depth = 0
    if args.modality == 'rgb':
        stack_depth = 1
    elif args.modality == 'flow':
        stack_depth = 5

    step = (frame_cnt - stack_depth) / (args.num_frame_per_video-1)
    if step > 0:
        frame_ticks = range(1, min((2 + step * (args.num_frame_per_video-1)), frame_cnt+1), step)
    else:
        frame_ticks = [1] * args.num_frame_per_video

    assert(len(frame_ticks) == args.num_frame_per_video)

    if args.modality == 'rgb':
        frame_list = []
        for tick in frame_ticks:
            name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            frame_list.append(frame)

        # run predication in batches
        scores = net.predict_rgb_frame_list(frame_list, score_name, frame_size=(340, 256))

    if args.modality == 'flow':
        flow_stack_list = []
        for tick in frame_ticks:
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]

            flow_stack = []
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))

            flow_stack_list.append(np.array(flow_stack))
        # run predication in batches
        scores = net.predict_flow_stack_list(flow_stack_list, score_name, frame_size=(340, 256))

    print 'video {} done'.format(vid)
    sys.stdin.flush()
    return scores, label



video_scores = map(eval_video, eval_video_list)

video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores]
video_labels = [x[1] for x in video_scores]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit/cls_cnt

print cls_acc

print 'Accuracy {:.02f}%'.format(np.mean(cls_acc)*100)

if args.save_scores is not None:
    np.savez(args.save_scores, scores=video_scores, labels=video_labels)




