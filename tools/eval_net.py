import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix

sys.path.append('.')
sys.path.append('/home/yjxiong/caffe/python')
from pyActionRecog import parse_directory, build_split_list
from pyActionRecog import parse_split_file
from pyActionRecog.action_caffe import CaffeNet
from pyActionRecog.utils.video_funcs import default_aggregation_func

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('split', type=int, choices=[1, 2, 3],
                    help='on which split to test the network')
parser.add_argument('modality', type=str, choices=['rgb', 'flow'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('net_proto', type=str)
parser.add_argument('net_weights', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument('--num_frame_per_video', type=int, default=25,
                    help="prefix of y direction flow images")
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--num_worker', type=int, default=1)
args = parser.parse_args()


# build neccessary information
print args.dataset
split_tp = parse_split_file(args.dataset)
f_info = parse_directory(args.frame_path,
                         args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix)


eval_video_list = split_tp[args.split - 1][1]


def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] if args.num_worker > 1 else 1
    net = CaffeNet(args.net_proto, args.net_weights, my_id-1)


def eval_video(video):
    global net
    label = video[1]
    vid = video[0]

    video_frame_path = f_info[0][vid]
    frame_cnt = f_info[1][vid]
    frame_ticks = range(1, frame_cnt+1, int(math.floor(frame_cnt / args.num_frame_per_video)))

    frame_scores = []
    for tick in frame_ticks:
        if args.modality == 'rgb':
            name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            scores = net.predict_single_frame([frame,], 'fc-action')
            frame_scores.append(scores)
        if args.modality == 'flow':
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(5)]
            flow_stack = []
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))
            scores = net.predict_single_flow_stack(np.array(flow_stack), 'fc-action')
            frame_scores.append(scores)

    print 'video {} done'.format(vid)
    sys.stdin.flush()
    return np.array(frame_scores), label

if args.num_worker > 1:
    pool = multiprocessing.Pool(args.num_worker, initializer=build_net)
    video_scores = pool.map(eval_video, eval_video_list)
else:
    build_net()
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




