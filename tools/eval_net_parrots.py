import argparse
import os
import sys
import math
import cv2
import numpy as np
import multiprocessing
from sklearn.metrics import confusion_matrix
import time

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
parser.add_argument('parrots_model', type=str)
parser.add_argument('parrots_weights', type=str)
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
parser.add_argument('--num_frame_per_video', type=int, default=25,
                    help="prefix of y direction flow images")
parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
parser.add_argument('--score_name', type=str, default='fc_action')
parser.add_argument('--num_worker', type=int, default=1)
parser.add_argument('--max_num_gpu', type=int, default=8)
parser.add_argument('--display', type=int, default=1)
parser.add_argument('--gpu_list', type=int, nargs='+', default=None)
parser.add_argument("--parrots_path", type=str, default='/home/yjxiong/Parrots/parrots/python',
                    help='path to the Parrots toolbox')
args = parser.parse_args()

print args

sys.path.append(args.parrots_path)
from pyActionRecog.action_parrots import ParrotsNet


def build_net(device_ids=None):
    global net

    my_id = multiprocessing.current_process()._identity[0] if args.num_worker > 1 else 1

    if device_ids:
        gpu_id = device_ids[my_id - 1]
    else:
        gpu_id = (my_id - 1) % args.max_num_gpu

    net = ParrotsNet(args.parrots_model, args.parrots_weights, gpu_id,
                     10, score_name=args.score_name)


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
        frame_scores = []
        for tick in frame_ticks:
            name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
            frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
            frame_scores.append(net.predict_single_rgb_frame(frame, score_name, frame_size=(340, 256)))

    if args.modality == 'flow':
        frame_scores = []
        for tick in frame_ticks:
            frame_idx = [min(frame_cnt, tick+offset) for offset in xrange(stack_depth)]

            flow_stack = []
            for idx in frame_idx:
                x_name = '{}{:05d}.jpg'.format(args.flow_x_prefix, idx)
                y_name = '{}{:05d}.jpg'.format(args.flow_y_prefix, idx)
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, x_name), cv2.IMREAD_GRAYSCALE))
                flow_stack.append(cv2.imread(os.path.join(video_frame_path, y_name), cv2.IMREAD_GRAYSCALE))

            frame_scores.append(net.predict_single_flow_stack(flow_stack, score_name, frame_size=(340, 256)))

    sys.stdin.flush()
    return np.array(frame_scores), label, vid


def callback(rst):
    global proc_start_time
    eval_rst.append(rst)
    cnt = len(eval_rst)
    if cnt % args.display == 0:
        cnt_time = time.time() - proc_start_time
        print 'video {} done, total {}/{}, average {} sec/video'.format(rst[-1], cnt,
                                                                        len(eval_video_list),
                                                                        float(cnt_time) / cnt)


if args.num_worker > 1:
    global proc_start_time
    pool = multiprocessing.Pool(args.num_worker, initializer=build_net, initargs=(args.gpu_list,))
    eval_rst = []
    proc_start_time = time.time()
    jobs = [pool.apply_async(eval_video, args=(x,), callback=callback) for x in eval_video_list]
    pool.close()
    pool.join()

    rst_dict = {x[-1]:x[:2] for x in eval_rst}
    video_scores = [rst_dict[v[0]] for v in eval_video_list]
else:
    build_net(0)
    eval_rst = []
    for i, v in enumerate(eval_video_list):
        eval_rst.append(eval_video(v))
        print 'video {} done, total {}/{}'.format(v[0], i, len(eval_video_list))
        # print eval_rst[-1][0][0, 0, :]

    video_scores = [x[:2] for x in eval_rst]

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




