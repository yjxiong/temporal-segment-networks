import argparse
import glob
import os
import fnmatch
import random

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('frame_path', type=str, help="root directory holding the frames")
parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x')
parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y')
parser.add_argument('--num_split', type=int, default=3)
parser.add_argument('--out_list_path', type=str, default='data/')
parser.add_argument('--shuffle', action='store_true', default=False)

args = parser.parse_args()

working_dataset = args.dataset
frame_path = args.frame_path
rgb_p = args.rgb_prefix
flow_x_p = args.flow_x_prefix
flow_y_p = args.flow_y_prefix
num_split = args.num_split
out_path = args.out_list_path
shuffle =  args.shuffle


def parse_directory(path):

    print 'parse frames under folder {}'.format(path)
    frame_folders = glob.glob(os.path.join(path, '*'))

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list] 
        return cnt_list

    # check RGB
    rgb_counts = {}
    flow_counts = {}
    dir_dict = {}
    for i,f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_p, flow_x_p, flow_y_p))
        k = f.split('/')[-1]
        rgb_counts[k] = all_cnt[0]
        dir_dict[k] = f

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        flow_counts[k] = x_cnt
        if i % 200 == 0:
            print '{} videos parsed'.format(i)

    print 'frame folder analysis done'
    return dir_dict, rgb_counts, flow_counts

split_parsers = {}
def parse_split_file(dataset):
    sp = split_parsers[dataset]
    return sp()

def build_split_list(split_tuple, frame_info, split_idx):
    split = split_tuple[split_idx]

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            frame_dir = frame_info[0][item[0]]
            rgb_cnt = frame_info[1][item[0]]
            flow_cnt = frame_info[2][item[0]]
            rgb_list.append('{} {} {}\n'.format(frame_dir, rgb_cnt, item[1]))
            flow_list.append('{} {} {}\n'.format(frame_dir, flow_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)


## Dataset specific split file parse
def parse_ucf_splits():
    class_ind = [x.strip().split() for x in open('data/ucf101_splits/classInd.txt')]
    class_mapping = {x[1]:x[0] for x in class_ind}

    def line2rec(line):
        items = line.strip().split('/')
        label = class_mapping[items[0]]
        vid = items[1].split('.')[0]
        return vid, label

    splits = []
    for i in xrange(1, 4):
        train_list = [line2rec(x) for x in open('data/ucf101_splits/trainlist{:02d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open('data/ucf101_splits/testlist{:02d}.txt'.format(i))]
        splits.append((train_list, test_list))
    return splits

split_parsers['ucf101'] = parse_ucf_splits
  
# operation
print 'processing dataset {}'.format(working_dataset)
split_tp = parse_split_file(working_dataset)
f_info = parse_directory(frame_path)

print 'writting list files for training/testing'
for i in xrange(num_split):
    lists = build_split_list(split_tp, f_info, i)
    open(os.path.join(out_path, '{}_rgb_train_split_{}.txt'.format(working_dataset, i+1)), 'w').writelines(lists[0][0])
    open(os.path.join(out_path, '{}_rgb_val_split_{}.txt'.format(working_dataset, i+1)), 'w').writelines(lists[0][1])
    open(os.path.join(out_path, '{}_flow_train_split_{}.txt'.format(working_dataset, i+1)), 'w').writelines(lists[1][0])
    open(os.path.join(out_path, '{}_flow_val_split_{}.txt'.format(working_dataset, i+1)), 'w').writelines(lists[1][1])
